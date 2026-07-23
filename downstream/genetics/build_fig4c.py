#!/usr/bin/env python3
"""Reproducibly build the Figure 4c locus->domain classification.

For each of the 18 LeDXA-specific lead SNPs we collect GWAS Catalog
associations for (a) the lead SNP itself and (b) every 1000G-EUR LD proxy at
r2 >= R2_MIN, then map each catalogued trait to an organ-system domain using an
explicit keyword table. A locus is tagged with the union of domains implied by
its lead + proxy associations. This replaces the previously hard-coded
LOCUS_DOMAINS dict so panel c is derived from auditable variant-level evidence.

Outputs (in this directory):
  fig4c_associations.tsv  - every lead/proxy association retrieved (cache)
  fig4c_locus_domains.tsv - one row per (locus, domain) with supporting evidence
  fig4c_domain_counts.tsv - domain -> number of loci
  fig4c_unmapped_traits.txt - traits that matched no domain (for manual review)
"""
import json, os, sys, time, subprocess

R2_MIN = 0.8
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'tables', 'fig4c'))
ASSOC_TSV = os.path.join(HERE, "fig4c_associations.tsv")

# (lead rsID, gene label as used in the figure)  -- GRCh37 leads
LOCI = [
    ("rs3753841",  "COL11A1"),
    ("rs12028481", "PRRX1"),
    ("rs12105038", "AC010096.2"),
    ("rs13417165", "AC007319.1"),
    ("rs1562502",  "ARPP21"),
    ("rs7433808",  "ADAMTS9-AS2"),
    ("rs61749613", "VCAN"),
    ("rs62432291", "FNDC1"),
    ("rs2057932",  "RSBN1L-TMEM60"),
    ("rs7816345",  "AC090453.1"),
    ("rs12548347", "8q12"),
    ("rs2370759",  "EPC1"),
    ("rs2274224",  "PLCE1-AS1"),
    ("rs7949030",  "GANAB"),
    ("rs585522",   "PITPNM2"),
    ("rs17184313", "RIN3"),
    ("rs35874463", "SMAD3"),
    ("rs6123685",  "BMP7"),
]

UA = {"User-Agent": "ledxa-fig4c/1.0", "Accept": "application/json"}


def _get(url, tries=4):
    # Shell out to curl: the uv-installed Python lacks a CA bundle, but curl
    # uses the system trust store and reaches both APIs fine.
    for i in range(tries):
        try:
            out = subprocess.run(
                ["curl", "-s", "--max-time", "40",
                 "-H", "Accept: application/json",
                 "-H", "User-Agent: ledxa-fig4c/1.0", url],
                capture_output=True, text=True, timeout=60)
            if out.returncode == 0 and out.stdout.strip():
                return json.loads(out.stdout)
            time.sleep(1.0 * (i + 1))
        except json.JSONDecodeError:
            return None
        except Exception:
            time.sleep(1.0 * (i + 1))
    return None


def gwas_assocs(rsid):
    """Return list of (trait, pvalue_str) for a variant, [] if none/404."""
    url = ("https://www.ebi.ac.uk/gwas/rest/api/singleNucleotidePolymorphisms/"
           f"{rsid}/associations?projection=associationBySnp&size=1000")
    js = _get(url)
    out = []
    if not js or "_embedded" not in js:
        return out
    for a in js["_embedded"].get("associations", []):
        pval = a.get("pvalue")
        if pval is None and a.get("pvalueMantissa") is not None:
            pval = f"{a['pvalueMantissa']}e{a['pvalueExponent']}"
        traits = [t.get("trait") for t in a.get("efoTraits", []) if t.get("trait")]
        if not traits and a.get("traitName"):
            traits = [a["traitName"]]
        for tr in traits:
            out.append((tr, str(pval)))
    return out


def ld_proxies(rsid, r2min=R2_MIN):
    url = (f"https://rest.ensembl.org/ld/human/{rsid}/1000GENOMES:phase_3:EUR"
           f"?content-type=application/json;r2={r2min}")
    js = _get(url)
    if not isinstance(js, list):
        return []
    prox = []
    for row in js:
        other = row["variation2"] if row["variation1"] == rsid else row["variation1"]
        prox.append((other, float(row["r2"])))
    # keep the highest r2 if a proxy appears twice
    best = {}
    for p, r2 in prox:
        if p.startswith("rs") and p != rsid:
            best[p] = max(best.get(p, 0.0), r2)
    return sorted(best.items(), key=lambda x: -x[1])


def fetch():
    rows = [("lead_snp", "gene", "assoc_snp", "assoc_type", "r2", "trait", "p_value")]
    for rsid, gene in LOCI:
        sys.stderr.write(f"[lead] {rsid} {gene}\n"); sys.stderr.flush()
        for tr, p in gwas_assocs(rsid):
            rows.append((rsid, gene, rsid, "lead", "1.0", tr, p))
        time.sleep(0.2)
        prox = ld_proxies(rsid)
        sys.stderr.write(f"    {len(prox)} proxies r2>={R2_MIN}\n"); sys.stderr.flush()
        for p, r2 in prox:
            assocs = gwas_assocs(p)
            for tr, pv in assocs:
                rows.append((rsid, gene, p, "proxy", f"{r2:.3f}", tr, pv))
            time.sleep(0.15)
    with open(ASSOC_TSV, "w") as fh:
        for r in rows:
            fh.write("\t".join(r) + "\n")
    sys.stderr.write(f"wrote {len(rows)-1} association rows -> {ASSOC_TSV}\n")


# ---- domain keyword table (lowercased substring match on the trait string) ----
DOMAIN_RULES = {
    "Body composition": [
        "fat mass", "fat-free", "fat free", "lean mass", "lean body mass",
        "appendicular", "body fat", "waist", "hip circumference", "hip index",
        "whr", "waist-hip", "waist-to-hip", "body mass index", "adipos",
        "visceral", "subcutaneous", "obesity", "body shape", "trunk fat",
        "body size", "predicted mass", "whole body", "weight", "fat pad",
        "anthropometric", "body surface area", "metabolic rate", "bioimpedance",
    ],
    "Bone": [
        "bone mineral density", "bone density", "bone area", "bone size",
        "heel bone", "femoral neck", "trochanter", "osteoporosis", "fracture",
        "paget", "sclerostin", "skeletal system", "estimated bone",
        "hip geometry", "bone mineral",
    ],
    "Height/Anthropometric": ["height"],
    "Cardiovascular": [
        "blood pressure", "pulse pressure", "pulse rate", "heart rate",
        "coronary", "atrial fibrillation", "cardiac", "cardiovascular",
        "hypertens", "myocardial", "heart failure", "electrocardiogra",
        "qt interval", "qrs", "aortic", "aneurysm", "arterial pressure",
        "stroke", "thromboemb", "artery", "vascular", "valve", "atrial",
        "ventric",
    ],
    "Renal": ["glomerular filtration", "kidney", "renal", "creatinine",
              "cystatin", "urate", "chronic kidney", "urea", "nephro"],
    "Pulmonary": ["forced expiratory", "fev", "fvc", "lung function", "pulmonary",
                  "bronchodilator", "copd", "asthma", "respiratory function",
                  "spirometry", "emphysema", "vital capacity", "airflow"],
    "Metabolic/Lipid": [
        "cholesterol", "hdl", "ldl", "triglyceride", "apolipoprotein",
        "lipoprotein", "type 2 diabetes", "type ii diabetes", "diabetes",
        "glucose", "hba1c", "glycated", "fatty acid", "metabolite",
        "metabolic syndrome", "phospholipid", "cholesteryl", "glycine",
        "amino acid", "igf-1", "insulin-like growth",
    ],
    "Neuro/Behavioral": [
        "neuroticism", "risk toler", "risk-taking", "risky", "cognit",
        "insomnia", "schizophren", "depress", "worry", "intelligence",
        "educational attainment", "smoking", "alcohol", "sleep",
        "wellbeing", "well-being", "reaction time", "mood", "anxiety",
        "externalizing", "mathematical", "brain", "cerebral", "cortical",
        "cortex", "white matter", "migraine", "headache", "tinnitus",
        "attention deficit", "tourette", "socioeconomic", "physical activity",
        "substance", "neuroimaging", "language measurement", "social inhibition",
    ],
    "Reproductive/Endocrine": [
        "sex hormone", "shbg", "testosterone", "menarche", "menopause",
        "estradiol", "oestradiol", "sexual dysfunction", "vitamin d",
        "hydroxyvitamin", "age at first", "thyroid", "endometriosis",
        "uterine", "leiomyoma", "breast", "menstrua",
    ],
    "Hepatic": ["alanine aminotransferase", "aspartate aminotransferase",
                "liver", "hepatic", "gamma glutamyl", "gamma-glutamyl",
                "bilirubin", "alkaline phosphatase", "albumin"],
    "Joint/connective-tissue": [
        "joint hypermobility", "hypermobility", "beighton", "carpal tunnel",
        "hernia", "osteoarthritis", "scoliosis", "connective tissue",
        "dupuytren", "intervertebral disc", "disc herniation", "spondyl",
        "arthroplasty", "dysplasia of the hip",
    ],
    "Immune/Hematologic": [
        "blood cell", "leukocyte", "lymphocyte", "monocyte", "eosinophil",
        "neutrophil", "basophil", "platelet", "haemoglobin", "hemoglobin",
        "haematocrit", "hematocrit", "erythrocyte", "reticulocyte",
        "white blood cell", "red blood cell", "immune", "autoimmune",
        "granulocyte", "multiple sclerosis", "psoriasis", "allerg", "eczema",
        "dermatitis", "crohn", "ulcerative colitis", "rheumatoid",
        "cholangitis", "inflammatory bowel",
    ],
    "Cancer": ["cancer", "carcinoma", "neoplasm", "tumor", "tumour", "melanoma",
               "glioma", "leukemia", "leukaemia", "lymphoma"],
    "Protein/molecular level": ["protein level", "levels of", "level of",
                                "protein quantit", "measurement in blood",
                                "amount of", "protein amount", "osteopontin",
                                "versican", "wnt inhibitory", "dentin matrix",
                                "collagen alpha", "serine protease",
                                "endothelial cell-specific"],
}


def classify():
    assoc = []
    with open(ASSOC_TSV) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            assoc.append(dict(zip(header, line.rstrip("\n").split("\t"))))

    unmapped = set()

    def domains_for(trait):
        t = trait.lower()
        ds = [d for d, kws in DOMAIN_RULES.items() if any(k in t for k in kws)]
        if not ds and trait not in ("(none catalogued)", ""):
            unmapped.add(trait)
        return ds

    # locus -> domain -> list of (assoc_snp, assoc_type, r2, trait, p)
    loc_dom = {}
    loc_gene = {}
    for a in assoc:
        lead = a["lead_snp"]; loc_gene[lead] = a["gene"]
        for d in domains_for(a["trait"]):
            loc_dom.setdefault(lead, {}).setdefault(d, []).append(
                (a["assoc_snp"], a["assoc_type"], a["r2"], a["trait"], a["p_value"]))

    # write per-(locus,domain) evidence
    with open(os.path.join(HERE, "fig4c_locus_domains.tsv"), "w") as fh:
        fh.write("lead_snp\tgene\tdomain\tn_assoc\tbest_evidence\n")
        for rsid, gene in LOCI:
            doms = loc_dom.get(rsid, {})
            if not doms:
                fh.write(f"{rsid}\t{gene}\t(none)\t0\t-\n")
                continue
            for d in sorted(doms):
                ev = doms[d]
                # prefer a lead-SNP evidence row for the label, else top proxy
                ev_sorted = sorted(ev, key=lambda e: (e[1] != "lead", -float(e[2])))
                s = ev_sorted[0]
                tag = "lead" if s[1] == "lead" else f"proxy {s[0]} r2={s[2]}"
                fh.write(f"{rsid}\t{gene}\t{d}\t{len(ev)}\t{s[3]} (P={s[4]}; {tag})\n")

    # domain counts (loci per domain)
    counts = {}
    for rsid in loc_dom:
        for d in loc_dom[rsid]:
            counts[d] = counts.get(d, 0) + 1
    with open(os.path.join(HERE, "fig4c_domain_counts.tsv"), "w") as fh:
        fh.write("domain\tn_loci\n")
        for d, n in sorted(counts.items(), key=lambda x: -x[1]):
            fh.write(f"{d}\t{n}\n")

    with open(os.path.join(HERE, "fig4c_unmapped_traits.txt"), "w") as fh:
        for t in sorted(unmapped):
            fh.write(t + "\n")

    print("=== domain counts (loci per domain) ===")
    for d, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {n:2d}  {d}")
    print(f"\n=== per-locus domains ===")
    for rsid, gene in LOCI:
        ds = ", ".join(sorted(loc_dom.get(rsid, {}))) or "(none)"
        print(f"  {gene:14s} {rsid:12s} -> {ds}")
    print(f"\n{len(unmapped)} unmapped trait strings -> fig4c_unmapped_traits.txt")


# on-target (musculoskeletal / body-composition) domains -- the paper's focus
ONTARGET = ("Bone", "Body composition", "Height/Anthropometric")
# pQTL / biomarker-level rows are not a phenotype domain for primary assignment
EXCLUDE_PRIMARY = ("Protein/molecular level",)


def _pf(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return 1.0


def primary_domains():
    """One primary domain per locus.

    Rule: use the LEAD SNP's own catalogued associations if it has any (only
    fall back to LD proxies when the lead is uncatalogued). Within the chosen
    evidence set, pick the strongest (lowest-P) on-target {Bone / Body
    composition / Height} association; if none is on-target, pick the strongest
    association overall. pQTL/biomarker-level rows are ignored for the label.
    """
    rows = []
    with open(ASSOC_TSV) as fh:
        h = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            rows.append(dict(zip(h, line.rstrip("\n").split("\t"))))

    def cands(pool):
        out = []  # (domain, p, snp, assoc_type, r2, trait)
        for a in pool:
            t = a["trait"].lower()
            for d, kws in DOMAIN_RULES.items():
                if d in EXCLUDE_PRIMARY:
                    continue
                if any(k in t for k in kws):
                    out.append((d, _pf(a["p_value"]), a["assoc_snp"],
                                a["assoc_type"], a["r2"], a["trait"]))
        return out

    with open(os.path.join(HERE, "fig4c_primary.tsv"), "w") as fh:
        fh.write("lead_snp\tgene\tprimary_domain\tevidence_snp\tassoc_type\tr2\t"
                 "trait\tp_value\tsource\n")
        print("\n=== PRIMARY domain per locus (lead-first) ===")
        counts = {}
        for rsid, gene in LOCI:
            lead = [a for a in rows if a["lead_snp"] == rsid and a["assoc_type"] == "lead"]
            prox = [a for a in rows if a["lead_snp"] == rsid and a["assoc_type"] == "proxy"]
            lead_c = cands(lead)
            src = "lead"
            pool_c = lead_c
            if not lead_c:
                pool_c = cands(prox)
                src = "proxy (lead uncatalogued)"
            if not pool_c:
                fh.write(f"{rsid}\t{gene}\t(none/novel)\t-\t-\t-\t-\t-\tno catalogued association\n")
                print(f"  {gene:14s} -> (none/novel)")
                continue
            ont = [c for c in pool_c if c[0] in ONTARGET]
            pick = min(ont if ont else pool_c, key=lambda c: c[1])
            d, p, snp, atype, r2, trait = pick
            counts[d] = counts.get(d, 0) + 1
            fh.write(f"{rsid}\t{gene}\t{d}\t{snp}\t{atype}\t{r2}\t{trait}\t{p}\t{src}\n")
            tag = "lead" if atype == "lead" else f"proxy r2={r2}"
            print(f"  {gene:14s} -> {d:22s} [{trait}, P={p:.0e}, {tag}]")
        print("\n=== primary-domain counts ===")
        for d, n in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {n:2d}  {d}")


if __name__ == "__main__":
    if "--classify-only" not in sys.argv and not os.path.exists(ASSOC_TSV):
        fetch()
    elif "--refetch" in sys.argv:
        fetch()
    classify()
    primary_domains()
