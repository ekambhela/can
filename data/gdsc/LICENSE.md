# Data licensing & attribution — `data/gdsc/`

The files in this directory are **third-party data**, redistributed here for
**research and educational use only**. They are *not* covered by the project's
MIT license (see the repository `LICENSE`). Each source retains its own terms.

## GDSC (Genomics of Drug Sensitivity in Cancer)

GDSC1 and GDSC2 drug-response and genomic-feature data are produced by the
**Wellcome Sanger Institute** and released for **academic use**. See
<https://www.cancerrxgene.org>.

If you use this data, cite:

- Iorio F. *et al.* "A Landscape of Pharmacogenomic Interactions in Cancer."
  *Cell* 166, 740–754 (2016).
- Yang W. *et al.* "Genomics of Drug Sensitivity in Cancer (GDSC): a resource
  for therapeutic biomarker discovery in cancer cells." *Nucleic Acids Research*
  41, D955–D961 (2013).

## Redistribution paths used here

- **GDSC1 (release 17)** matrices (`IC50_v17.csv.gz`,
  `genomic_features_v17.csv.gz`, `drug_decode_gdsc.csv`) come from the official
  Sanger **`gdsctools`** Python package, which is licensed **BSD-3-Clause** and
  redistributes these release-17 tables.
- **`drug_list_gdsc.csv`** (GDSC1 compound annotation) was obtained via the
  public `kimmo1019/DeepCDR` repository, validated against the `gdsctools`
  drug-decode export (all anchor drug ids match).
- **GDSC2 (25 Feb 2020)** fitted-dose-response data (`IC50_gdsc2.csv.gz`,
  `drug_list_gdsc2.csv`) was derived from the `GDSC2_fitted_dose_response_25Feb20`
  table redistributed in the public `jianglikun/DeepTTC` repository.

The underlying measurements are GDSC's in every case; the intermediate
repositories are redistribution channels only. See `README.md` in this
directory for the per-file schema.
