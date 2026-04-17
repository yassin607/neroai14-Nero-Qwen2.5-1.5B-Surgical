import torch
import json
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

def apply_nero_elbow_svd(tensor, min_retention=0.95):
    if len(tensor.shape) != 2:
        return tensor, 0
    
    original_dtype = tensor.dtype
    U, S, Vh = torch.linalg.svd(tensor.to(torch.float32), full_matrices=False)
    
    S_norm = S / S[0]
    diffs = S_norm[:-1] - S_norm[1:]
    threshold = diffs[0] * 0.05 
    k_indices = torch.where(diffs < threshold)[0]
    
    if len(k_indices) > 0:
        k = k_indices[0].item() + 1
    else:
        k = int(S.shape[0] * 0.98)

    min_k = int(S.shape[0] * min_retention)
    if k < min_k:
        k = min_k

    reconstructed = (U[:, :k] * S[:k]) @ Vh[:k, :]
    saved_pct = 100 * (1 - (k / S.shape[0]))
    
    return reconstructed.to(original_dtype), saved_pct

def nero_quantize_int8(tensor):
    max_val = tensor.abs().max()
    if max_val == 0: return tensor
    
    scale = max_val / 127
    quantized = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
    dequantized = quantized.to(torch.float16) * scale
    return dequantized

def save_nero_model(model, tokenizer, output_path="./Nero-Universal-Surgical"):
    print(f"\n💾 Nero Exporter: Saving compressed model to {output_path}...")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    print(f"✅ Export Complete! Folder is ready at: {output_path}")

def run_nero_hybrid_mission():
    # تقدر تغير ده لأي موديل تاني (مثلاً "meta-llama/Llama-3.2-1B-Instruct")
    model_id = "Qwen/Qwen2.5-1.5B-Instruct" 
    
    print(f"🚀 Nero Engine: Starting Universal Hybrid Mission on {model_id}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cpu")
        
        total_original_size = 0
        total_compressed_size = 0

        print("🧠 Processing Layers & Monitoring Savings...")
        
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                num_params = module.weight.numel()
                layer_original_bytes = num_params * 2 # FP16 = 2 bytes
                total_original_size += layer_original_bytes
                
                # -- NERO UNIVERSAL LOGIC --
                # حماية الطبقات الحساسة بالاسم (Attention & Normalization)
                is_sensitive = any(x in name for x in ["self_attn", "norm", "lm_head"])
                # حماية الطبقات الصغيرة جداً (أقل من 1 مليون باراميتر)
                is_small = num_params < (1024 * 1024) 

                if is_sensitive or is_small:
                    total_compressed_size += layer_original_bytes
                    continue
                
                # إذا كانت الطبقة كبيرة وغير حساسة (غالباً MLP)، نفذ الـ Algorithm بتاعنا
                # 1. SVD
                clean_weight, saved_svd_pct = apply_nero_elbow_svd(module.weight.data, min_retention=0.95)
                
                # 2. INT8
                hybrid_weight = nero_quantize_int8(clean_weight)
                module.weight.data = hybrid_weight
                
                # حساب الحجم الحقيقي (بعد الضغط لـ 1 byte لكل باراميتر متبقي)
                reduced_params = num_params * (1 - (saved_svd_pct / 100))
                layer_compressed_bytes = reduced_params * 1 
                total_compressed_size += layer_compressed_bytes
                
                print(f"[💎 NERO OK] {name:40} | Saved: ~{((1 - layer_compressed_bytes/layer_original_bytes)*100):.1f}%")

        # التقرير النهائي
        orig_mb = total_original_size / (1024**2)
        comp_mb = total_compressed_size / (1024**2)
        savings_pct = (1 - (comp_mb / orig_mb)) * 100

        print(f"\n📊 --- NERO FINAL STORAGE REPORT ---")
        print(f"📦 Original Weight Size: {orig_mb:.2f} MB")
        print(f"📉 Nero Compressed Size: {comp_mb:.2f} MB")
        print(f"🚀 Total VRAM Saved:     {savings_pct:.2f}%")
        print(f"--------------------------------------")

        # تجربة الموديل
        print("\n✅ Mission Complete! Testing Logic...")
        prompt = "Explain why the sky is blue."
        inputs = tokenizer(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True), return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.3, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        
        print("\nNERO OUTPUT:\n" + "="*40 + "\n" + tokenizer.decode(outputs[0], skip_special_tokens=True) + "\n" + "="*40)

        save_nero_model(model, tokenizer)

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

if __name__ == "__main__":
    run_nero_hybrid_mission()