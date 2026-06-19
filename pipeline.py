"""
PIPELINE PRINCIPAL  Modelo Hbrido de LLM
===========================================
Orquestra as 10 etapas do fluxo completo:

  1. Recepo do prompt
  2. Extrao de conceitos [L1]
  3. Refinamento por Juzos Kantianos [L2]
  4. Silogismo Cientfico + Hempel
  5. Falseabilidade de Popper
  6. Avaliao Paraconsistente [L3]
  7. Sntese por Equivalncia [L4]
  8. Gerao da Resposta [L5  opcional]
  9. Resposta Final em Texto Fluida [L6]
 10. Texto Final Definitivo [L7]

Usa config_loader, knowledge_base (KB escalvel + RAG opcional), l5_generation
e opcionalmente o agente de pesquisa para enriquecer contexto.
"""

from __future__ import annotations
import sys
import re
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch

from llm_provider_client import DEFAULT_MODELS, SUPPORTED_PROVIDERS, normalize_provider
from neural_truth_model import TruthScoringModel, load_tokenizer
from l1_concept_table import ConceptTable, ConceptNode, LogicLMSymbolicSolver
from l2_kantian_judgments import KantianJudgmentEngine, KantianJudgment
from syllogism_module import ScientificSyllogismPipeline
from l3_paraconsistent import ParaconsistentEngine, ParaconsistentValue
from l4_synthesis import RussellianSynthesisEngine, SynthesisResult
from l6_final_response import EpistemicContext, FinalResponseEngine
from l7_final_text import FinalTextEngine
from cot_hierarchical import HierarchicalCoTTrace
from layer_titles import LAYER_TITLES

try:
    from l4_russell_equivalence import load_concept_base
except Exception:
    load_concept_base = None  # type: ignore

try:
    from config_loader import load_config, PROJECT_ROOT
except Exception:
    load_config = None  # type: ignore
    PROJECT_ROOT = Path(__file__).resolve().parent

try:
    from knowledge_base import get_knowledge_base, SEED_KNOWLEDGE_BASE
except Exception:
    get_knowledge_base = None  # type: ignore
    SEED_KNOWLEDGE_BASE = {}

try:
    from l5_generation import generate_response as l5_generate
except Exception:
    l5_generate = None  # type: ignore

try:
    from agente_busca_web import run_search_for_context
except Exception:
    run_search_for_context = None  # type: ignore

try:
    from agente_sintese_final import synthesize_final_text as synthesize_final_agent
except Exception:
    synthesize_final_agent = None  # type: ignore


def _get_kb(config: Optional[Dict[str, Any]], prompt: str, use_agent: bool) -> Dict[str, float]:
    if get_knowledge_base is None:
        return dict(SEED_KNOWLEDGE_BASE) if SEED_KNOWLEDGE_BASE else {}
    return get_knowledge_base(
        config=config,
        query_for_rag=prompt if use_agent else None,
    )


def _ensure_l1_config(config: Dict[str, Any]) -> Dict[str, Any]:
    l1_cfg = config.setdefault("l1", {})
    if not isinstance(l1_cfg, dict):
        l1_cfg = {}
        config["l1"] = l1_cfg
    l1_cfg.setdefault("spacy_enabled", True)
    languages = l1_cfg.get("spacy_languages", ["pt", "en"])
    if not isinstance(languages, list):
        languages = ["pt", "en"]
    l1_cfg["spacy_languages"] = [lang for lang in languages if lang in {"pt", "en"}] or ["pt", "en"]
    return config


def _resolve_provider_settings(
    config: Optional[Dict[str, Any]],
    section: str,
    default_provider: str = "ollama",
) -> Dict[str, Any]:
    """Normaliza configuraes de provider e escolhe o modelo correto para a etapa."""
    section_cfg = (config or {}).get(section, {}) if isinstance(config, dict) else {}
    provider = normalize_provider(section_cfg.get("provider") or default_provider)
    if provider not in SUPPORTED_PROVIDERS:
        provider = normalize_provider(default_provider)

    explicit_model = (section_cfg.get("model") or "").strip()
    ollama_model = (section_cfg.get("ollama_model") or explicit_model or "doninha8:latest").strip()
    base_url = (section_cfg.get("base_url") or "").strip()
    api_key = (section_cfg.get("api_key") or "").strip()
    ollama_host = (section_cfg.get("ollama_host") or "http://localhost:11434").strip()

    if provider == "ollama":
        resolved_model = ollama_model
    elif provider in {"template", "custom_lm"}:
        resolved_model = explicit_model or ollama_model
    else:
        resolved_model = explicit_model or DEFAULT_MODELS.get(provider, ollama_model)

    return {
        "provider": provider,
        "model": explicit_model,
        "resolved_model": resolved_model,
        "base_url": base_url,
        "api_key": api_key,
        "ollama_model": ollama_model,
        "ollama_host": ollama_host,
    }


class HybridLLMPipeline:
    """
    Pipeline completo do Modelo Hbrido de LLM.
    Suporta config, KB escalvel, L5 (gerao), agente opcional e chat.
    """

    def __init__(
        self,
        knowledge_base: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ) -> None:
        self._config = _ensure_l1_config(config or (load_config() if load_config else {}))
        self.kb = knowledge_base or _get_kb(self._config, "", False)
        if not self.kb:
            self.kb = dict(SEED_KNOWLEDGE_BASE) if SEED_KNOWLEDGE_BASE else {}
        self.verbose = verbose

        self.L1 = ConceptTable()
        self.L2 = KantianJudgmentEngine(self.L1)
        self.SYL = ScientificSyllogismPipeline()

        # L3
        l3_cfg = self._config.get("l3", {})
        model_path = l3_cfg.get("model_path", "truth_scoring_model.pt")
        backbone_name = l3_cfg.get("backbone", "bert-base-multilingual-cased")
        if not Path(model_path).is_absolute():
            model_path = str(PROJECT_ROOT / model_path)
        neural_model = None
        neural_tokenizer = None
        if os.path.exists(model_path):
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                neural_tokenizer = load_tokenizer(backbone_name)
                neural_model = TruthScoringModel(backbone_name=backbone_name)
                state = torch.load(model_path, map_location=device)
                neural_model.load_state_dict(state)
                neural_model.to(device)
                if self.verbose:
                    print(f"[L3] Modelo neural carregado de '{model_path}'")
                self.L3 = ParaconsistentEngine(neural_model=neural_model, neural_tokenizer=neural_tokenizer, device=device)
            except Exception as exc:
                if self.verbose:
                    print(f"[L3] Falha ao carregar modelo neural: {exc}")
                self.L3 = ParaconsistentEngine()
        else:
            self.L3 = ParaconsistentEngine()

        # L4
        russell_base = None
        rpath = self._config.get("l4", {}).get("russell_concepts_path", "l4_russell_concepts.json")
        if not Path(rpath).is_absolute():
            rpath = str(PROJECT_ROOT / rpath)
        if load_concept_base and os.path.exists(rpath):
            try:
                russell_base = load_concept_base(rpath)
                if self.verbose:
                    print("[L4] Base russelliana carregada.")
            except Exception:
                pass
        if russell_base is None and load_concept_base:
            try:
                from l4_russell_equivalence import build_russell_concept_base
                russell_base = build_russell_concept_base()
            except Exception:
                pass
        self.L4 = RussellianSynthesisEngine(
            self.kb,
            russell_concept_base=russell_base,
            use_concept_based_weights=(russell_base is not None),
            verification_config=self._config.get("l4_chain_verification", {}),
        )
        self.L6 = FinalResponseEngine()
        self.L7 = FinalTextEngine(config=self._config)  # Passa config para suportar mltiplos providers

    def _infer_domain(self, concepts: List[ConceptNode]) -> str:
        """Inferncia simples de domnio majoritrio a partir dos conceitos extrados."""
        if not concepts:
            return "geral"
        domain_counts = {}
        for concept in concepts:
            domain = concept.domain.lower().strip() if concept.domain else "geral"
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        return max(domain_counts, key=domain_counts.get)

    def _collect_canonical_sources(self, concepts: List[ConceptNode]) -> List[str]:
        sources = []
        seen = set()
        for concept in concepts:
            source = (concept.canonical_source or "").strip()
            if source and source not in seen:
                seen.add(source)
                sources.append(source)
        return sources

    def _summarize_judgments(self, judgments: List[KantianJudgment]) -> str:
        parts = []
        for idx, judgment in enumerate(judgments[:5], start=1):
            cls = getattr(judgment.epistemic_classification, "classification", "no_classificado")
            truth = getattr(judgment.epistemic_classification, "truth", 0.0)
            ind = getattr(judgment.epistemic_classification, "indeterminacy", 0.0)
            fals = getattr(judgment.epistemic_classification, "falsity", 0.0)
            parts.append(
                f"L2-{idx}: {judgment.proposicao[:120]} | pri={judgment.prioridade:.2f} | class={cls} | T/I/F={truth:.2f}/{ind:.2f}/{fals:.2f}"
            )
        return " ; ".join(parts) if parts else "nenhum juzo L2 disponvel"

    def _summarize_paraconsistent(self, pv_list: List[ParaconsistentValue]) -> str:
        parts = []
        for idx, pv in enumerate(pv_list[:5], start=1):
            ensemble_bits = []
            if getattr(pv, "confidence", None) is not None:
                ensemble_bits.append(f"conf={pv.confidence:.3f}")
            if getattr(pv, "heuristic_weight", None) is not None and getattr(pv, "neural_weight", None) is not None:
                ensemble_bits.append(f"h={pv.heuristic_weight:.3f}/n={pv.neural_weight:.3f}")
            if getattr(pv, "ensemble_agreement", None) is not None:
                ensemble_bits.append(f"agreement={pv.ensemble_agreement:.3f}")
            ensemble_summary = f" | ensemble {' '.join(ensemble_bits)}" if ensemble_bits else ""
            parts.append(
                f"L3-{idx}: ={pv.mu:.3f} ={pv.lam:.3f} state={pv.state} truth={pv.truth_value:.3f} certainty={pv.certainty:+.3f} contradiction={pv.contradiction:+.3f}{ensemble_summary}"
            )
        return " ; ".join(parts) if parts else "nenhuma avaliao L3 disponvel"

    def _build_citation_note(self, concepts: List[ConceptNode], agent_context: str) -> str:
        sources = self._collect_canonical_sources(concepts)
        if sources:
            return (
                "Fontes locais identificadas no contexto, mas nenhuma citao bibliogrfica externa foi confirmada para esta resposta; "
                "a exibio de referncias exige verificao direta da consulta ao documento."
            )
        if agent_context:
            return "Contexto externo detectado, mas nenhuma citao bibliogrfica foi confirmada para esta resposta."
        return "Nenhuma citao bibliogrfica foi confirmada para esta resposta; o resultado foi produzido com base interna e sem referncia externa verificada."

    def _append_audit_block(self, text: str, label: str, details: str) -> str:
        block = f"\n\n[AUDIT {label}] {details}"
        return (text + block).strip()

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        return (time.perf_counter() - start) * 1000

    @staticmethod
    def _clip(text: str, limit: int = 420) -> str:
        text = " ".join((text or "").split())
        return text[:limit] + "..." if len(text) > limit else text

    def process(
        self,
        prompt: str,
        chat_session: Optional[Any] = None,
        use_agent: Optional[bool] = None,
        skip_l5: bool = False,
        skip_l6: bool = False,
        return_cot: bool = False,
    ) -> SynthesisResult:
        """Executa o pipeline e retorna SynthesisResult (com response j gerada por L5 se ativo)."""
        t0 = time.perf_counter()
        cot_trace = HierarchicalCoTTrace(prompt=prompt)
        if use_agent is False:
            use_agent = True
        use_agent = use_agent if use_agent is not None else self._config.get("agent", {}).get("use_agent", True)
        if use_agent is False:
            use_agent = True
        if chat_session and hasattr(chat_session, "get_context_for_prompt"):
            prompt_for_kb = chat_session.get_context_for_prompt(prompt, self._config.get("chat", {}).get("max_turns_in_context", 10))
        else:
            prompt_for_kb = prompt

        # KB pode ser enriquecido por RAG (Chroma) quando use_agent
        if use_agent and get_knowledge_base:
            self.kb = _get_kb(self._config, prompt_for_kb, True)
            if not self.kb:
                self.kb = dict(SEED_KNOWLEDGE_BASE) if SEED_KNOWLEDGE_BASE else {}

        self._log("\n" + "" * 60)
        self._log(f"  PROMPT: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
        self._log("" * 60)

        limit = RussellianSynthesisEngine.check_fundamental_limits(prompt)
        if limit:
            self._log(f"\n{limit}")

        self._log("\n[ETAPA 2] L1  Extrao de Conceitos")
        layer_start = time.perf_counter()
        concepts: List[ConceptNode] = self.L1.extract_concepts(prompt, llm_context=prompt_for_kb, domain="geral", config=self._config)
        domain = self._infer_domain(concepts)
        if domain != "geral":
            # Re-extrai com domnio especfico para enriquecer com KB do domnio
            concepts = self.L1.extract_concepts(prompt, llm_context=prompt_for_kb, domain=domain, config=self._config)
        concepts_summary = ""
        if self.verbose and concepts:
            for c in concepts:
                syns = ", ".join(c.synonyms[:2]) or ""
                self._log(f"   {c.term:15s} | sinnimos: {syns}")
            concepts_summary = "; ".join(f"{c.term}({', '.join(c.synonyms[:2])})" for c in concepts[:8])
        cot_trace.add_step(
            "L1",
            LAYER_TITLES["l1"],
            "Extracao semantica de conceitos, dominio e relacoes de aplicacao a partir do prompt e do contexto disponivel.",
            [
                f"{len(concepts)} conceitos identificados",
                f"Dominio inferido: {domain}",
            ],
            concepts_summary or "Nenhum conceito estruturado encontrado.",
            self._elapsed_ms(layer_start),
        )

        self._log("\n[ETAPA 3] L2  Juzos Kantianos")
        layer_start = time.perf_counter()
        judgments: List[KantianJudgment] = self.L2.refine(prompt, concepts)
        top_judgments = ""
        if judgments:
            top_judgments = "\n".join(j.proposicao for j, _ in list(zip(judgments, [None] * 6))[:6])
        cot_trace.add_step(
            "L2",
            LAYER_TITLES["l2"],
            "Construcao de proposicoes epistemologicas, classificacao kantiana e priorizacao para as etapas formais.",
            [
                f"{len(judgments)} juizos gerados",
                f"Maior prioridade: {judgments[0].prioridade:.2f}" if judgments else "Sem juizos priorizados",
            ],
            self._clip(top_judgments, 520) or "Nenhum juizo gerado.",
            self._elapsed_ms(layer_start),
        )

        self._log("\n[ETAPAS 4+5] Silogismo + Hempel + Popper")
        prompt_terms = set(re.findall(r"[a-zA-Z]+", prompt.lower()))
        kb_scores = {j.proposicao[:30]: self.kb.get(j.proposicao.split()[0], 0.3) for j in judgments}
        filtered = self.SYL.run(judgments, prompt_terms, kb_scores)
        self._log(f"  {len(judgments)} hipteses  {len(filtered)} aps filtros")

        self._log("\n[ETAPA 6] L3  Lgica Paraconsistente + Classificao Epistemolgica L2")
        layer_start = time.perf_counter()
        props_with_priority = [(j.proposicao, score) for j, score in filtered]
        pv_list: List[ParaconsistentValue] = self.L3.evaluate(props_with_priority, self.kb)
        consistent = self.L3.check_global_consistency(pv_list)
        self._log(f"  Consistncia global: {'' if consistent else ''}")
        l3_cot_summary = self._summarize_paraconsistent(pv_list)
        cot_trace.add_step(
            "L3",
            LAYER_TITLES["l3"],
            "Avaliacao paraconsistente com mu/lambda, estados logicos e pesos dinamicos do ensemble quando disponiveis.",
            [
                f"Consistencia global: {'sim' if consistent else 'nao'}",
                f"{len(set(pv.state for pv in pv_list))} estados logicos distintos" if pv_list else "Nenhum estado logico produzido",
            ],
            self._clip(l3_cot_summary, 620),
            self._elapsed_ms(layer_start),
        )

        epistemic_context = EpistemicContext(
            proposition_states=[
                {
                    "proposition": pv.proposition,
                    "proposition_type": pv.proposition_kind or "Desconhecido",
                    "mu": pv.mu,
                    "lambda": pv.lam,
                    "certainty": pv.certainty,
                    "contradiction": pv.contradiction,
                    "truth_value": pv.truth_value,
                    "state": pv.state,
                    "confidence": pv.confidence,
                    "heuristic_weight": pv.heuristic_weight,
                    "neural_weight": pv.neural_weight,
                    "ensemble_agreement": pv.ensemble_agreement,
                    "neural_state": pv.neural_state,
                    "neural_truth": pv.neural_truth,
                }
                for pv in pv_list
            ],
            many_valued_routes=[
                {
                    "left": left.proposition,
                    "left_type": left.proposition_kind or "Desconhecido",
                    "right": right.proposition,
                    "right_type": right.proposition_kind or "Desconhecido",
                    "route": route,
                    "confidence": confidence,
                    "explanation": explanation,
                }
                for left, right, route, confidence, explanation in self.L3.route_contradictions(pv_list)
            ],
            bert_classifications=[
                {
                    "proposition": judgment.proposicao,
                    "priority": judgment.prioridade,
                    "truth": judgment.epistemic_classification.truth,
                    "indeterminacy": judgment.epistemic_classification.indeterminacy,
                    "falsity": judgment.epistemic_classification.falsity,
                    "classification": judgment.epistemic_classification.classification,
                }
                for judgment, _ in filtered[:8]
            ],
            application_context=LogicLMSymbolicSolver.summarize_application_context(concepts),
        )

        self._log("\n[ETAPA 7] L4  Sntese Russelliana")
        layer_start = time.perf_counter()
        l2_priorities = {j.proposicao[:40]: j.prioridade for j, _ in filtered}
        result: SynthesisResult = self.L4.synthesize(pv_list, l2_priorities, prompt)
        l4_result = result
        l5_text = result.response
        cot_trace.add_step(
            "L4",
            LAYER_TITLES["l4"],
            "Sintese por equivalencia russelliana e verificacao CoVe sobre a melhor hipotese L3 ponderada por prioridade L2.",
            [
                f"Estado sintetico: {result.state}",
                f"Valor-verdade: {result.truth_value:.3f}",
                f"Verificacoes CoVe: {len(getattr(result, 'verification_log', []) or [])}",
            ],
            self._clip(result.response, 560),
            self._elapsed_ms(layer_start),
        )

        # Contexto do agente (busca web/local) apenas quando o pipeline está em modo de fallback heurístico.
        agent_context = ""
        heuristic_only_mode = (
            (not self.kb or not concepts or not pv_list)
            or any(
                getattr(pv, "state", "") in {"Indeterminado", "Intermedirio", "Inconsistente_local"}
                or getattr(pv, "truth_value", 0.0) < 0.45
                or getattr(pv, "neural_weight", 0.0) == 0.0
                for pv in pv_list
            )
        )
        if use_agent and run_search_for_context and heuristic_only_mode:
            try:
                agent_context = run_search_for_context(prompt)
                if agent_context and self.verbose:
                    self._log("\n[AGENTE] Contexto de busca obtido para fallback heurístico.")
            except Exception:
                pass

        # L4  nota de fontes e auditoria (agora com contexto RAG disponvel)
        l4_sources_note = self._build_citation_note(concepts, agent_context)
        l4_result.response = self._append_audit_block(
            l4_result.response,
            "L4",
            f"truth={l4_result.truth_value:.4f} certainty={l4_result.certainty:+.4f} contradiction={l4_result.contradiction:+.4f} state={l4_result.state} | {l4_sources_note}"
        )
        result.response = l4_result.response
        l5_text = result.response

        # L5  Gerao de resposta em texto livre
        gen_cfg = self._config.get("generation", {})
        final_cfg = self._config.get("finalization", {})
        l7_cfg = self._config.get("l7", {})

        gen_resolved = _resolve_provider_settings(self._config, "generation")
        final_resolved = _resolve_provider_settings(self._config, "finalization")
        l7_resolved = _resolve_provider_settings(self._config, "l7")

        base_provider = gen_resolved["provider"]

        if final_cfg.get("provider") and normalize_provider(final_cfg.get("provider")) != base_provider:
            self._log(
                f"[PIPELINE] Ignorando provider de finalizao '{final_cfg.get('provider')}' para usar provider base '{base_provider}'."
            )
        if l7_cfg.get("provider") and normalize_provider(l7_cfg.get("provider")) != base_provider:
            self._log(
                f"[PIPELINE] Ignorando provider L7 '{l7_cfg.get('provider')}' para usar provider base '{base_provider}'."
            )

        provider = base_provider
        if not skip_l5 and l5_generate and provider != "template":
            layer_start = time.perf_counter()
            final_response = l5_generate(
                prompt,
                result,
                provider=provider,
                concepts_summary=concepts_summary,
                top_judgments=top_judgments,
                custom_lm_path=gen_cfg.get("custom_lm_path", ""),
                ollama_model=gen_resolved["resolved_model"],
                ollama_host=gen_resolved["ollama_host"],
                base_url=gen_resolved["base_url"],
                api_key=gen_resolved["api_key"],
            )
            if agent_context and final_response:
                final_response = final_response + "\n\n[Contexto da busca]\n" + agent_context[:800]
            elif agent_context:
                final_response = result.response + "\n\n[Contexto da busca]\n" + agent_context[:800]
            else:
                final_response = final_response or result.response
            result = SynthesisResult(
                response=self._append_audit_block(
                    final_response,
                    "L5",
                    f"provider={provider} model={gen_resolved['resolved_model']} | audit=L1-L5: {self._summarize_judgments(judgments)}"
                ),
                truth_value=result.truth_value,
                certainty=result.certainty,
                contradiction=result.contradiction,
                state=result.state,
                supporting_evidence=result.supporting_evidence,
                falsified_hypotheses=result.falsified_hypotheses,
                confidence_label=result.confidence_label,
            )
            cot_trace.add_step(
                "L5",
                LAYER_TITLES["l5"],
                "Geracao textual intermediaria a partir da sintese L4 e dos resumos L1-L3.",
                [
                    f"Provider: {provider}",
                    f"Modelo: {gen_resolved['resolved_model']}",
                ],
                self._clip(final_response, 560),
                self._elapsed_ms(layer_start),
            )
        elif agent_context and result.response:
            layer_start = time.perf_counter()
            result = SynthesisResult(
                response=self._append_audit_block(
                    result.response + "\n\n[Contexto da busca]\n" + agent_context[:800],
                    "L5",
                    f"provider={provider} model={gen_resolved['resolved_model']} | audit=L1-L5: {self._summarize_judgments(judgments)}"
                ),
                truth_value=result.truth_value,
                certainty=result.certainty,
                contradiction=result.contradiction,
                state=result.state,
                supporting_evidence=result.supporting_evidence,
                falsified_hypotheses=result.falsified_hypotheses,
                confidence_label=result.confidence_label,
            )
            cot_trace.add_step(
                "L5",
                LAYER_TITLES["l5"],
                "Geracao textual intermediaria em modo fallback, anexando contexto de busca quando disponivel.",
                [f"Provider: {provider}", "Fallback textual usado"],
                self._clip(result.response, 560),
                self._elapsed_ms(layer_start),
            )
        else:
            cot_trace.add_step(
                "L5",
                LAYER_TITLES["l5"],
                "Camada de geracao textual manteve a resposta sintetica de L4 como fallback.",
                [f"Provider: {provider}", "Sem chamada gerativa adicional"],
                self._clip(result.response, 560),
                0.0,
            )

        l5_text = result.response

        if not skip_l6:
            layer_start = time.perf_counter()
            final_text = self.L6.finalize_response(
                prompt=prompt,
                synthesis_result=result,
                epistemic_context=epistemic_context,
                generated_text=result.response,
                concepts_summary=concepts_summary,
                top_judgments=top_judgments,
                agent_context=agent_context,
            )
            final_text = self.L6.rewrite_response(
                prompt=prompt,
                synthesis_result=result,
                epistemic_context=epistemic_context,
                generated_text=final_text,
                concepts_summary=concepts_summary,
                top_judgments=top_judgments,
                agent_context=agent_context,
                provider=base_provider,
                custom_lm_path=final_cfg.get("custom_lm_path", gen_cfg.get("custom_lm_path", "")),
                ollama_model=final_resolved["resolved_model"],
                ollama_host=final_resolved["ollama_host"],
                base_url=final_resolved["base_url"],
                api_key=final_resolved["api_key"],
            )
            result = SynthesisResult(
                response=self._append_audit_block(
                    final_text,
                    "L6",
                    f"provider={base_provider} model={final_resolved['resolved_model']} | epistemic={self._summarize_paraconsistent(pv_list)}"
                ),
                truth_value=result.truth_value,
                certainty=result.certainty,
                contradiction=result.contradiction,
                state=result.state,
                supporting_evidence=result.supporting_evidence,
                falsified_hypotheses=result.falsified_hypotheses,
                confidence_label=result.confidence_label,
            )
            cot_trace.add_step(
                "L6",
                LAYER_TITLES["l6"],
                "Refinamento da resposta intermediaria para clareza, coerencia e proporcionalidade epistemica.",
                [
                    f"Provider: {base_provider}",
                    f"Modelo: {final_resolved['resolved_model']}",
                ],
                self._clip(final_text, 560),
                self._elapsed_ms(layer_start),
            )
        else:
            cot_trace.add_step(
                "L6",
                LAYER_TITLES["l6"],
                "Refinamento final foi pulado por configuracao da chamada.",
                ["skip_l6=True"],
                self._clip(result.response, 420),
                0.0,
            )

        l3_summary = ""
        if epistemic_context is not None and epistemic_context.proposition_states:
            top_states = epistemic_context.proposition_states[:3]
            l3_summary = "; ".join(
                f"{item.get('proposition', 'desconhecida')}  {item.get('state', 'n/a')} ({item.get('truth_value', 0):.2f}, conf={item.get('confidence', 0) or 0:.2f}, h/n={item.get('heuristic_weight', 0) or 0:.2f}/{item.get('neural_weight', 0) or 0:.2f})"
                for item in top_states
            )
            if epistemic_context.many_valued_routes:
                l3_summary += f"; rotas paraconsistentes: {len(epistemic_context.many_valued_routes)}"

        l7_cfg = self._config.get("l7", {})
        # Coletar alertas de incompatibilidade cannica gerados durante L1
        canonical_alerts = LogicLMSymbolicSolver.get_canonical_alerts() if LogicLMSymbolicSolver else []
        
        # === L7  Texto Final Definitivo (Automtico e Integrado) ===
        # Usa o agente de sntese final quando disponvel, com fallback para a engine L7 original.
        if synthesize_final_agent is not None:
            layer_start = time.perf_counter()
            final_text_l7 = synthesize_final_agent(
                prompt=prompt,
                l1_summary=concepts_summary,
                l2_summary=top_judgments,
                l3_summary=l3_summary,
                l4_response=l4_result.response,
                l5_text=l5_text,
                l6_text=result.response,
                provider=base_provider,
                model=l7_resolved["resolved_model"],
                temperature=l7_cfg.get("temperature", 0.7),
                max_tokens=l7_cfg.get("max_tokens", 4096),
                cot_context=cot_trace.to_markdown(),
            )
        else:
            layer_start = time.perf_counter()
            final_text_l7 = self.L7.finalize_text(
                prompt=prompt,
                l1_summary=concepts_summary,
                l2_summary=top_judgments,
                l3_summary=l3_summary,
                l4_response=l4_result.response,
                l5_text=l5_text,
                l6_text=result.response,
                synthesis_result=l4_result,
                provider=base_provider,
                model=l7_resolved["resolved_model"],
                custom_lm_path=l7_cfg.get("custom_lm_path", gen_cfg.get("custom_lm_path", "")),
                canonical_alerts=canonical_alerts,
                cot_context=cot_trace.to_markdown(),
                temperature=l7_cfg.get("temperature", 0.7),
                max_tokens=l7_cfg.get("max_tokens", 4096),
            )
        cot_trace.add_step(
            "L7",
            LAYER_TITLES["l7"],
            "Sintese final integra os resumos L1-L6 e o rastro auditavel para produzir a resposta definitiva.",
            [
                f"Provider: {base_provider}",
                f"Modelo: {l7_resolved['resolved_model']}",
            ],
            self._clip(final_text_l7, 620),
            self._elapsed_ms(layer_start),
        )
        cot_trace.final_synthesis = final_text_l7
        cot_trace.overall_confidence = float(getattr(result, "truth_value", 0.0) or 0.0)
        result = SynthesisResult(
            response=self._append_audit_block(
                final_text_l7,
                "L7",
                f"provider={base_provider} model={l7_resolved['resolved_model']} | sources={'; '.join(self._collect_canonical_sources(concepts)[:6]) or 'nenhuma fonte local'} | L2={self._summarize_judgments(judgments)} | L3={self._summarize_paraconsistent(pv_list)}"
            ),
            truth_value=result.truth_value,
            certainty=result.certainty,
            contradiction=result.contradiction,
            state=result.state,
            supporting_evidence=result.supporting_evidence,
            falsified_hypotheses=result.falsified_hypotheses,
            confidence_label=result.confidence_label,
        )
        result.cot_trace = cot_trace  # type: ignore[attr-defined]
        result.cot_markdown = cot_trace.to_markdown() if return_cot else ""  # type: ignore[attr-defined]

        elapsed = (time.perf_counter() - t0) * 1000
        self._log(f"\n[ETAPA 10] L7  Texto Final Definitivo  ({elapsed:.1f} ms)\n")
        self._log(str(result))
        return result

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def repl(self) -> None:
        print("\n" + "" * 60)
        print("  MODELO HBRIDO DE LLM  Fonseca")
        print("  Digite 'sair' para encerrar")
        print("" * 60)
        while True:
            try:
                prompt = input("\nPrompt  ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt:
                continue
            if prompt.lower() in {"sair", "exit", "quit"}:
                break
            self.process(prompt)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Modelo Hbrido de LLM  Pipeline L1L7")
    parser.add_argument("--prompt", "-p", type=str, help="Pergunta nica (imprime s a resposta)")
    parser.add_argument("--repl", action="store_true", help="Modo interativo")
    parser.add_argument("--demo", action="store_true", help="Rodar demonstrao com prompts fixos")
    parser.add_argument("--config", type=str, help="Caminho para config.yaml")
    args, _ = parser.parse_known_args()

    config = load_config(Path(args.config)) if load_config and args.config else (load_config() if load_config else {})
    pipeline = HybridLLMPipeline(config=config, verbose=not args.prompt)

    if args.prompt:
        r = pipeline.process(args.prompt)
        print(r.response)
        return
    if args.repl:
        pipeline.repl()
        return
    if args.demo:
        for p in ["A gua a 35 graus est quente ou fria?", "O que  a verdade?"]:
            pipeline.process(p)
            print()
        return
    # Default: demo + repl se --repl no argv antigo
    if "--repl" in sys.argv:
        pipeline.repl()
        return
    for p in ["A gua a 35 graus est quente ou fria?", "O que  a verdade?"]:
        pipeline.process(p)
        print()


if __name__ == "__main__":
    main()
