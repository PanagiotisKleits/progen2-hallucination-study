

#!/usr/bin/env python3
import time
import os
import re
import sys
import argparse
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────

PARAMS_TSV     = "progen2_params.tsv"
LENGTHS_TSV    = "protein_lengths.tsv"
OUTPUT_DIR     = "progen2_outputs2"
NUM_SAMPLES    = 100   # proteins per (t, p) combination




from tokenizers import Tokenizer





# ── Model loading ───────────────────────────────────────────────────────────────
def load_model(device,  checkpoint_path, ProGenForCausalLM,multi_gpu=False):
    print(f"Loading progen2-small from {checkpoint_path} ")
    model = ProGenForCausalLM.from_pretrained(checkpoint_path)
    model.eval()

    if multi_gpu and torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = torch.nn.DataParallel(model)

    model = model.to(device)
    return model


def load_tokenizer(progen2_dir):
    path = os.path.join(progen2_dir, "tokenizer.json")
    return Tokenizer.from_file(path)


# ── Generation ──────────────────────────────────────────────────────────────────
def generate_proteins(model, tokenizer, temperature, top_p, num_samples, device, repetition_penalty, length):
    start_id = tokenizer.encode("1").ids[0]
    end_id   = tokenizer.encode("2").ids[0]

    if batch_size is None:
        batch_size = num_samples 

    sequences = []
    remaining = num_samples # counter that tracks how many sequences still need to be generated.

    while remaining > 0:
        current_batch = min(batch_size, remaining) # generates either a full batch or whatever is left, whichever is smaller
        input_ids = torch.tensor([[start_id]] * current_batch).to(device) # creates a tensor with one start token for each sequence in the batch, and moves it to the device
        attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=length,
            eos_token_id=end_id,
            pad_token_id=end_id,
            repetition_penalty=repetition_penalty,
        )

    sequences = []
    for seq in output:
        decoded = tokenizer.decode(seq.tolist(), skip_special_tokens=True).strip()
        cleaned = ''.join(c for c in decoded if c in 'ACDEFGHIKLMNPQRSTVWY')
        if cleaned:
          sequences.append(cleaned)
    
    remaining -= current_batch
    
    return sequences


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",     default="cpu", choices=["cpu", "cuda"],help="'cpu' for 96-core VM, 'cuda' for GPU VMs")
    parser.add_argument("--multi-gpu",  action="store_true",help="Enable DataParallel across all available GPUs (7-GPU VM)")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES, help="Proteins to generate per (t, p) combination")
    parser.add_argument("--params-tsv", default=PARAMS_TSV)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--lengths-tsv", default=LENGTHS_TSV)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--progen2-dir", default="/bcpl_bcpl/Software/progen/progen/progen2")
    parser.add_argument("--checkpoints", default="/bcpl_vari/progen_DATA/progen/progen2/checkpoints/progen2-small")
    parser.add_argument("--batch-size", type=int, default=None, help="Sequences per model.generate() call. Defaults to all num_samples at once.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    sys.path.insert(0, args.progen2_dir)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    from models.progen.modeling_progen import ProGenForCausalLM
    from transformers import GenerationMixin

    if GenerationMixin not in ProGenForCausalLM.__bases__:
        ProGenForCausalLM.__bases__ = ProGenForCausalLM.__bases__ + (GenerationMixin,)

    # ── Device setup ───────────────────────────────────────────────────────────
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
            gpu_count = torch.cuda.device_count()
            print(f"GPU(s) available: {gpu_count}")
            for i in range(gpu_count):
                print(f"  [{i}] {torch.cuda.get_device_name(i)}")
    else:
        device = torch.device("cpu")
        n_threads = os.cpu_count()
        torch.set_num_threads(n_threads)
        print(f"CPU mode — using {n_threads} threads")

    # ── Load model once ────────────────────────────────────────────────────────
    model = load_model(device, args.checkpoints, ProGenForCausalLM=ProGenForCausalLM, multi_gpu=args.multi_gpu)
    tokenizer = load_tokenizer(args.progen2_dir)

    # ── Read (t, p) parameters ─────────────────────────────────────────────────
    params = []
    with open(args.params_tsv) as f:
        next(f)  # skip header
        for line in f:
            parts = re.split(r'\t+', line.strip())
            if len(parts) >= 2:
                t = float(parts[0])
                p = float(parts[1])
                params.append((t, p))

    # ── Read length and population ─────────────────────────────────────────────
    lengths = []
    with open(args.lengths_tsv) as f:
        next(f)  # skip header
        for line in f:
            parts = re.split(r'\t+', line.strip())
            if len(parts) >= 2:
                length = int(parts[0])
                num_samples = int(parts[1]) // 10
                lengths.append((length, num_samples))

    print(f"\nRunning {len(params)} parameter combinations × {len(lengths)} samples\n")

    all_rows = []
    total_start = time.time()

    progress_path = os.path.join(args.output_dir, "progress.txt") # builds the path to the progress file inside the output directory
    completed = set()
    if os.path.exists(progress_path): #checks if a progress file already exists from a previous run
        with open(progress_path) as pf:
            for line in pf:
                completed.add(line.strip()) # reads each line from the progress file and adds it to the completed set

    total_combinations = len(params) * len(lengths) #calculates the total number of combinations 
    completed_count = len(completed) # initialises the counter to however many combinations were already done in a previous run
    
    for t, p in params:
      for length, num_samples in lengths:

          run_id = f"t{t}_p{p}_len{length}"
          print(f"→ {run_id} ", end=" ", flush=True)

          sequences = generate_proteins(model, tokenizer, t, p,num_samples,device,args.repetition_penalty, length)

        # Save per-combination FASTA
          fasta_path = os.path.join(args.output_dir, f"all_sequences.fasta")
          with open(fasta_path, "a") as f:
             for i, seq in enumerate(sequences, 1):
                 f.write(f">{run_id}_seq{i}\n{seq}\n")

          print(f"done  ({len(sequences)} sequences → {fasta_path})")

    total_time=time.time()-total_start
    print(f"\nAll done! Results saved to {args.output_dir}/all_sequences.fasta")
    print(f"Total time: {total_time:.1f} sec")

if __name__ == "__main__":
    main()
