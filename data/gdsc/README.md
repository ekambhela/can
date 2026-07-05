# GDSC data (real)

Genomics of Drug Sensitivity in Cancer (GDSC) — two independent screens.

## GDSC1 (release 17)
- `IC50_v17.csv.gz` — natural-log IC50 for 988 human cancer cell lines across 265 screened compounds (`COSMIC_ID` + `Drug_<id>_IC50` columns).
- `genomic_features_v17.csv.gz` — per cell line: `TISSUE_FACTOR`, `MSI_FACTOR`, driver-gene mutation flags (`*_mut`) and copy-number alterations (`gain_*`, `loss_*`).
- `drug_decode_gdsc.csv` — authoritative drug id → name/target export from the GDSC database (9-drug subset, used to validate the full list below).
- `drug_list_gdsc.csv` — the full GDSC screened-compounds annotation (`drug_id`, `Name`, `Synonyms`, `Targets`, `Target pathway`, `PubCHEM`) covering 264 of the 265 v17 compounds. Obtained via the public DeepCDR repository (`kimmo1019/DeepCDR`), which redistributes the GDSC drug list; validated against `drug_decode_gdsc.csv` (all 9 anchor drug ids match exactly).

## GDSC2 (25 Feb 2020 release) — added screen
- `IC50_gdsc2.csv.gz` — natural-log IC50 from the GDSC2 assay for the 806 GDSC2 cell lines that are also in `genomic_features_v17` (`COSMIC_ID` + `Drug_<id>_IC50`). Drug ids are namespaced `200000 + GDSC2_drug_id` so they never collide with GDSC1 ids. Only the **126 GDSC2 compounds not present in the GDSC1 panel** are kept, so no drug appears twice.
- `drug_list_gdsc2.csv` — GDSC2 drug annotation (`drug_id`, `Name`, `Targets`, `Target pathway`) derived from the GDSC2 fitted-dose-response file's `PUTATIVE_TARGET` / `PATHWAY_NAME` columns.
- Source: the GDSC2 fitted-dose-response table (`GDSC2_fitted_dose_response_25Feb20`) redistributed in the public DeepTTC repository (`jianglikun/DeepTTC`). Both screens key on the same GDSC `COSMIC_ID`, so GDSC2 joins directly onto the existing genomic-feature matrix.

Source: the GDSC1 matrices come from the official Sanger `gdsctools` Python package (BSD-3); GDSC2 from the public GDSC2 fitted-dose-response release. GDSC is released for academic use — see https://www.cancerrxgene.org.

Cite: Iorio et al., *Cell* 2016; Yang et al., *Nucleic Acids Research* 2013.
