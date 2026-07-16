# sample_data/

Real DXA scans (UK Biobank, Human Phenotype Project) are access-controlled and are **not**
distributed with this repository. To let you verify the model runs without institutional data,
`demo.py` builds the LeDXA encoder and runs a forward pass on a **synthetic** random batch shaped
like a whole-body DXA scan (`384 × 128`):

```bash
pip install -e .
python sample_data/demo.py
# input (2, 3, 384, 128) -> embedding (2, 384)  (embed_dim=384)
```

Point the training/extraction code (`model/`) at your own DXA data to produce real embeddings.
