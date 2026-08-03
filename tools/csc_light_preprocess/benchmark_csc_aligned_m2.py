#!/usr/bin/env python3
"""Run the opt-in in-memory csc_aligned preprocessing sweep without exports."""
import argparse,csv,json,subprocess,tempfile
from pathlib import Path
from csc_light_preprocess import build_if_needed,load_txt_csc,write_binary_csc
MATRICES=("ASIC_100k","Stanford","bcsstk32","cant","consph","crankseg_2","ct20stif","lhr71","ohne2","pdb1HYS","pwtk","rma10","shipsec1","soc-sign-epinions","webbase-1M","xenon2")
POLICIES=("round_robin","load_only","load_similarity")
FIELDS=("matrix","policy","rows","columns","nnz","matrix_load_ms","feature_extraction_ms_median","mapping_ms_median","ordering_ms_median","fp32_conversion_ms","aligned_materialization_ms","metadata_generation_ms","aligned_validation_ms","aligned_preprocessing_total_ms","aligned_descriptor_count","aligned_useful_value_bytes","aligned_physical_value_bytes","aligned_value_padding_bytes","simd_utilization","bg_nnz_cv_aligned","bg_chunk_cv","lockstep_rounds","critical_bg","value_transactions","row_index_transactions","logical_mul_events")
def main():
 p=argparse.ArgumentParser();p.add_argument("--matrix-dir",type=Path,default=Path("sparse_matrix_csc"));p.add_argument("--output",type=Path,required=True);p.add_argument("--threads",type=int,default=1);p.add_argument("--only");a=p.parse_args()
 if a.output.exists():raise SystemExit(f"refusing overwrite: {a.output}")
 binary=Path(__file__).with_name("csc_light_preprocess_bin");build_if_needed(binary);rows=[]
 for name in MATRICES:
  if a.only and name!=a.only:continue
  nr,nc,cp,ri,val,load_ms=load_txt_csc(a.matrix_dir/f"{name}_csc.txt")
  with tempfile.NamedTemporaryFile(prefix="csc_m2_",suffix=".bin") as staged:
   write_binary_csc(Path(staged.name),nr,nc,cp,ri,val)
   for policy in POLICIES:
    cmd=[str(binary),"--input",staged.name,"--layout","csc_aligned","--policy",policy,"--num-bg","64","--choices","4","--segment-nnz","0","--sketch-bits","256","--alpha","1","--beta","0.1","--seed","1","--threads",str(a.threads),"--warmup","0","--repeat","1"]
    rec=json.loads(subprocess.run(cmd,check=True,text=True,capture_output=True).stdout);rec["matrix"]=name;rec["matrix_load_ms"]=load_ms;rows.append(rec)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open("x",newline="") as f:
  w=csv.DictWriter(f,fieldnames=FIELDS,extrasaction="ignore");w.writeheader();w.writerows({k:r.get(k,"") for k in FIELDS} for r in rows)
 print(a.output)
if __name__=="__main__":main()
