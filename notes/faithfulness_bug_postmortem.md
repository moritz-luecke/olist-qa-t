# Bug-Postmortem: Fragen-Leak im Temporal-Faithfulness-Judge

**Komponente:** `phase5_temporal/temporal_faithfulness.py` (LLM-Extraktions-Prompt)
**Schweregrad:** hoch — falsche Zahlen in der Headline-Faithfulness-Tabelle des Papers
**Status:** behoben (Metrik auf deterministisch/answer-only umgestellt)
**Entdeckt:** beim Erstellen des κ-Annotations-Templates (Schritt 2 der Kalibrierung),
als die zeitblinden Antworten fast durchweg „INSUFFICIENT EVIDENCE" waren, die
gemeldete Coverage aber bei 64–80 % lag.

---

## 1. Zusammenfassung

Die LLM-Judge-Variante der Temporal-Faithfulness-Metrik extrahierte die
„behaupteten" Zeiträume **aus der Frage statt aus der Antwort**. Dadurch wurden
sogar abstinierende Antworten („INSUFFICIENT EVIDENCE") als „covered" gewertet.
Die berichtete **Coverage war systematisch aufgebläht** und für **jedes** System
größer als mathematisch möglich. Die Accuracy-Zahlen waren **nicht** betroffen.

---

## 2. Symptom

Im stratifizierten 30-Fragen-Sample des Systems `entity` waren **alle** drift-
und **alle** latency-Fragen „INSUFFICIENT EVIDENCE" (≈ 28/30 ohne Periodenbezug,
also ≈ 7 % Coverage). Im Paper stand für `entity` aber **75.6 % Coverage**.

Gegenprüfung über alle 90 Antworten (Zählung von „INSUFFICIENT EVIDENCE"):

| System | INSUFFICIENT-Rate | max. mögliche Coverage | **Coverage im Paper** |
|---|---|---|---|
| vanilla_v2 | 71 % | ≤ 29 % | 80 % |
| entity | 78 % | ≤ 22 % | 75.6 % |
| community | **100 %** | **0 %** | 64.4 % |
| hybrid | 84 % | ≤ 16 % | 72.2 % |
| entity_temporal | 40 % | ≤ 60 % | 91.1 % |
| community_temporal | 26 % | ≤ 74 % | 95.6 % |
| hybrid_temporal | 26 % | ≤ 74 % | 95.6 % |

Wenn ein System bei *n %* der Fragen abstiniert (keine Periode nennt), ist
`coverage_ok` dort zwingend `false` → Coverage ≤ **(100 − INSUFFICIENT) %**.
**Jede** Paper-Coverage lag über dieser Decke. Eindeutigster Fall:
**community antwortet zu allen 90 Fragen „INSUFFICIENT" (0/90 Accuracy) — kann
also unmöglich 64.4 % Coverage haben.**

---

## 3. Smoking Gun

Datensatz `faithfulness_community_llm_reclassified.jsonl`, qid `trend_07`:

```
question:  "Compare the total number of orders placed in Q1 2017 versus Q4 2017."
answer:    "INSUFFICIENT EVIDENCE"
gold_anchors: ["2017-Q1", "2017-Q4"]
extracted:    ["2017-Q1", "2017-Q4"]    ← aus der FRAGE, nicht der Antwort
missing_gold: []
coverage_ok:  true                       ← obwohl die Antwort nichts behauptet
```

Über alle Systeme die Zahl der Fälle „Antwort = INSUFFICIENT, aber
`coverage_ok = true`": vanilla 47, entity 51, **community 58**, hybrid 55,
entity_temporal 31, community/hybrid_temporal 20.

---

## 4. Ursache

Der Extraktions-Prompt in `temporal_faithfulness.py` zeigte dem Judge **beide**
Blöcke:

```
QUESTION:
{question}
ANSWER:
{answer}
```

Die Aufgabe lautete „liste jeden Zeitraum, den die ANTWORT behauptet". Das LLM
zog die Perioden jedoch aus der **Frage** (die die Gold-Anker fast immer wörtlich
enthält). Ergebnis: Die Coverage-Metrik maß faktisch **„stehen die Gold-Anker in
der Frage?"** (fast immer ja) statt **„referenziert die Antwort sie?"**.

Nebeneffekt: Auch die **Halluzinations**-Zahl (~0 % überall) war kontaminiert —
die extrahierten Fragen-Perioden liegen per Konstruktion im erlaubten Fenster und
lösten nie einen Out-of-Window-Flag aus.

**Nicht betroffen:** die Accuracy/Match-Zahlen. Die kommen aus dem `match`-Feld
der `*_scored.jsonl` und haben mit dem Faithfulness-Judge nichts zu tun.

---

## 5. Fix (Option A — deterministisch, answer-only)

Die Metrik wurde **vollständig vom LLM-Judge gelöst** und deterministisch aus dem
**Antworttext** berechnet (kein Judge, kein Fragen-Leak, kein κ nötig):

- **Answer-Rate** = Anteil Antworten ≠ „INSUFFICIENT EVIDENCE".
- **Coverage** (konservativ) = Anteil, deren Text alle `gold_anchors` als exakten
  Perioden-/Monats-String nennt. Untergrenze (Paraphrasen wie „Q2 of 2017" werden
  nicht gutgeschrieben).
- **Halluzination** (unter Antworten) = Anteil mit einer Periode außerhalb des
  Gold-Fensters (+2 Monate Lifecycle-Spillover).

Reproduzierbar ohne API über `temporal_faithfulness.py --silver` (answer-only).

---

## 6. Vorher / Nachher

| System | Coverage **alt (Bug)** | Answer-Rate **neu** | Coverage **neu (kons.)** | Halluz. **neu** |
|---|---|---|---|---|
| vanilla_v2 | 80.0 % | 28.9 % | 12.2 % | 7.7 % |
| entity | 75.6 % | 22.2 % | 10.0 % | 10.0 % |
| community | 64.4 % | 0.0 % | 0.0 % | — |
| hybrid | 72.2 % | 15.6 % | 3.3 % | 0.0 % |
| entity_temporal | 91.1 % | 60.0 % | 36.7 % | 1.9 % |
| community_temporal | 95.6 % | 74.4 % | 41.1 % | 0.0 % |
| hybrid_temporal | 95.6 % | 74.4 % | 42.2 % | 0.0 % |

---

## 7. Auswirkung auf die Paper-Aussage

**Keine Änderung der Kernaussage — sie wurde ehrlicher.** Die Faithfulness-Story
verschob sich von „Coverage-Gap" zu **„Answer-Rate-Gap"**: der dominante
zeitblinde Failure-Modus ist **Abstention**, nicht Fabrikation. Das Paper trägt
weiterhin zwei harte Säulen:

1. **Accuracy:** D/E 60 % vs. bestes zeitblindes 11 % (**+48.9 pp**), trend 30/30.
2. **Answer-Rate:** D/E 74 % vs. zeitblind ≤ 29 % (community 0 %).

Die pre-registrierte Hypothese **H2** („zeitblind halluziniert ≥ 25 %") ist damit
**widerlegt** und wird im Paper transparent als widerlegte Pre-Registration plus
Reframe berichtet (nicht unterdrückt).

---

## 8. Betroffene Artefakte

**Verworfen (nicht mehr verwenden):**
`outputs/results/faithfulness_*_llm.jsonl` und
`outputs/results/faithfulness_*_llm_reclassified.jsonl` (alle 7 Systeme).

**Im Paper korrigiert:** Abstract, Intro (RQ4 + Beitrag 3 + Findings),
§3 Method (Faithfulness-Beschreibung), §5 Experiments (Metrik-Liste, keine
Judge-Kalibrierung, Cost-Tabelle: Faithfulness-Judge $25 → $0), §6 Results
(Faithfulness-Tabelle + Subsection neu), §8 Limitations (H2-Reframe,
„judge-free by design"), §9 Ethics (ein Judge statt zwei), Appendix
(Judge-Prompt → deterministische Metrik-Beschreibung + Notiz zum verworfenen
Judge).

---

## 9. Lessons Learned

- **Sanity-Check via Decke:** Eine Metrik gegen ihre mathematische Obergrenze
  prüfen (Coverage ≤ 100 − Abstention). Hätte den Bug sofort aufgedeckt.
- **Frage und Antwort trennen:** Bei „extrahiere X aus der Antwort"-Aufgaben darf
  der Prompt die Frage nicht mitliefern, sonst leakt sie in die Extraktion.
- **Deterministisch vor LLM-Judge:** Wo eine Metrik mechanisch berechenbar ist
  (String-/Perioden-Match), ist das robuster, reproduzierbar und nicht von
  Prompt-Artefakten abhängig.
- **Der ehrliche Befund war stärker:** Die Abstention-Rate ist eine klarere,
  unangreifbarere Geschichte als die fragile LLM-Coverage.

---

## 10. Verifikation

```bash
# Abstention/Answer-Rate (rein deterministisch):
#   Anteil Antworten, die nicht "INSUFFICIENT EVIDENCE" sind, je System
# Coverage + Halluzination answer-only (bug-frei):
python phase5_temporal/temporal_faithfulness.py --silver \
  --answers outputs/results/temporal_questions_90_<sys>_answers.jsonl \
  --questions outputs/temporal_questions_90.jsonl
```

Die deterministischen Zahlen aus §6 sind allein aus dem `answer`-Feld der
`temporal_questions_90_<sys>_answers.jsonl` reproduzierbar — keine API nötig.
