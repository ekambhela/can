# GDSC data (real)

Genomics of Drug Sensitivity in Cancer (GDSC), release 17.

- `IC50_v17.csv.gz` — natural-log IC50 for 988 human cancer cell lines across 265 screened compounds (`COSMIC_ID` + `Drug_<id>_IC50` columns).
- `genomic_features_v17.csv.gz` — per cell line: `TISSUE_FACTOR`, `MSI_FACTOR`, driver-gene mutation flags (`*_mut`) and copy-number alterations (`gain_*`, `loss_*`).
- `drug_decode_gdsc.csv` — authoritative drug id → name/target export from the GDSC database (subset).

Source: the official Sanger `gdsctools` Python package (BSD-3), which redistributes these GDSC example matrices. GDSC is released for academic use — see https://www.cancerrxgene.org.

Cite: Iorio et al., *Cell* 2016; Yang et al., *Nucleic Acids Research* 2013.
