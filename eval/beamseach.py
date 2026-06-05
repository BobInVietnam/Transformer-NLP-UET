import torch
import torch.nn.functional as F
from tqdm import tqdm

def generate_and_decode_batch_beam_search(
    model, dataloader, tgt_tokenizer, device, beam_size=3, max_gen_len=50, alpha=0.6
):
    """
    Runs batch-parallelized Beam Search decoding across a dataloader to generate text predictions.
    
    Args:
        model: The trained unified Transformer model.
        dataloader: The validation or test DataLoader.
        tgt_tokenizer (Vocabulary): Your custom target language Vocabulary helper instance.
        device: The compute hardware target ('cuda' or 'cpu').
        beam_size (int): Number of parallel structural tracking paths (B).
        max_gen_len (int): Maximum token generation limit per sequence.
        alpha (float): Length normalization penalty coefficient.
        
    Returns:
        generated_sentences (list of str): The decoded strings predicted by the model.
        reference_sentences (list of str): The decoded strings of the true ground truth labels.
    """
    model.eval()
    
    # Extract special token mappings directly from your custom Vocabulary class instance
    sos_id = tgt_tokenizer.stoi["<SOS>"]
    eos_id = tgt_tokenizer.stoi["<EOS>"]
    pad_id = tgt_tokenizer.stoi["<PAD>"]
    
    generated_sentences = []
    reference_sentences = []
    
    progress_bar = tqdm(dataloader, desc="Generating Sequences (Beam Search)", leave=True)
    
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].unsqueeze(1).unsqueeze(2).to(device)
            labels = batch["labels"]  # Kept on CPU for string decoding mappings
            
            batch_size = input_ids.size(0)
            
            # Step 1: Pre-compute Encoder states
            encoder_output = model.encoder(x=input_ids, mask=attention_mask)
            
            # Step 2: Loop independently over each sample item inside the batch
            for i in range(batch_size):
                # Isolate the current item context from the batch
                single_encoder_output = encoder_output[i : i + 1]  # Keep dimension shape (1, src_seq_len, d_model)
                single_attention_mask = attention_mask[i : i + 1]  # Keep dimension shape (1, 1, 1, src_seq_len)
                
                # Each lane tracks a tuple: (cumulative_log_probability, token_id_list)
                beams = [(0.0, [sos_id])]
                completed_beams = []
                
                # Autoregressive generation sequence loop
                for step in range(max_gen_len):
                    candidates = []
                    
                    # Package all active paths into a unified input matrix
                    beam_inputs = [tokens for _, tokens in beams]
                    dec_input = torch.tensor(beam_inputs, dtype=torch.long, device=device)
                    current_beam_size = dec_input.size(0)
                    
                    # Replicate single sample context vectors to match active parallelized beams
                    b_encoder_output = single_encoder_output.repeat(current_beam_size, 1, 1)
                    b_src_mask = single_attention_mask.repeat(current_beam_size, 1, 1, 1)
                    
                    # Compute Decoder scores
                    logits = model.decoder(x=dec_input, encoder_output=b_encoder_output, src_mask=b_src_mask)
                    
                    # Isolate log probs for the last predicted token step
                    next_token_logits = logits[:, -1, :]
                    log_probs = F.log_softmax(next_token_logits, dim=-1)
                    
                    # Pull top B choices to reduce unnecessary matrix iteration
                    topk_log_probs, topk_ids = torch.topk(log_probs, beam_size, dim=-1)
                    
                    # Expand active track lanes
                    for b_idx in range(current_beam_size):
                        cum_log_prob, tokens = beams[b_idx]
                        
                        for k in range(beam_size):
                            next_token_prob = topk_log_probs[b_idx, k].item()
                            next_token_id = topk_ids[b_idx, k].item()
                            
                            new_cum_prob = cum_log_prob + next_token_prob
                            new_tokens = tokens + [next_token_id]
                            
                            # Route path to completed pool if it hits <EOS>
                            if next_token_id == eos_id:
                                completed_beams.append((new_cum_prob, new_tokens))
                            else:
                                candidates.append((new_cum_prob, new_tokens))
                                
                    # Early termination check
                    if len(completed_beams) >= beam_size:
                        break
                        
                    # Filter down back to the top B lanes globally
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    beams = candidates[:beam_size]
                    
                    if not beams:
                        break
                        
                # Fallback dump if no lane hit an explicit <EOS> boundary
                if not completed_beams:
                    completed_beams = beams
                    
                # Step 3: Apply Length Normalization adjustment penalty
                normalized_beams = []
                for log_prob, tokens in completed_beams:
                    length = len(tokens) - 1  # Deduct starting <SOS> from measurement
                    length = max(length, 1)
                    norm_score = log_prob / (length ** alpha)
                    normalized_beams.append((norm_score, tokens))
                    
                # Isolate the highest scoring track candidate
                normalized_beams.sort(key=lambda x: x[0], reverse=True)
                best_tokens = normalized_beams[0][1]
                
                # Step 4: Final array cropping and text decoding string mapping
                pred_tokens = best_tokens[1:]  # Crop out initial <SOS>
                if eos_id in pred_tokens:
                    pred_tokens = pred_tokens[:pred_tokens.index(eos_id)]
                    
                # Use your Vocabulary.itos helper dictionary to map index positions back to strings
                pred_words = [tgt_tokenizer.itos.get(idx, "<UNK>") for idx in pred_tokens]
                pred_string = " ".join(pred_words)
                generated_sentences.append(pred_string)
                
                # Unpack, clean up, and decode corresponding dataset references via itos map
                true_tokens = [idx for idx in labels[i].tolist() if idx not in (pad_id, sos_id, eos_id)]
                true_words = [tgt_tokenizer.itos.get(idx, "<UNK>") for idx in true_tokens]
                true_string = " ".join(true_words)
                reference_sentences.append(true_string)
                
    return generated_sentences, reference_sentences

def generate_summary_beam_search(model, article_text, src_vocab, tgt_vocab, device, beam_size=3, max_len=50, alpha=0.6):
    """
    Generates a text summary from an input article using parallelized Beam Search decoding.

    Args:
        model: Your trained top-level unified Transformer model.
        article_text (str): The raw input source sentence/article string.
        src_vocab (Vocabulary): The source language Vocabulary helper instance.
        tgt_vocab (Vocabulary): The target language Vocabulary helper instance.
        device (torch.device): Active hardware compute target ('cuda' or 'cpu').
        beam_size (int): Number of alternative tracking paths to maintain (B).
        max_len (int): Maximum token generation threshold cutoff.
        alpha (float): Length normalization penalty coefficient.

    Returns:
        str: The final decoded summary sentence.
    """
    model.eval()

    # 1. Look up exact special token IDs dynamically from your Vocabulary maps
    pad_id = tgt_vocab.stoi["<PAD>"]
    sos_id = tgt_vocab.stoi["<SOS>"]
    eos_id = tgt_vocab.stoi["<EOS>"]

    # 2. Tokenize and numericalize the raw input article string using your helper
    src_tokens = article_text.lower().split()
    src_ids = src_vocab.numericalize(src_tokens)

    # 3. Convert to a 2D batch tensor of shape (1, src_seq_len) -> Batch Size = 1
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

    # 4. Generate the 4D Encoder padding mask using your pad_id value
    src_mask = (src_tensor != pad_id).unsqueeze(1).unsqueeze(2).to(device)

    with torch.no_grad():
        # Pre-compute the Encoder context once to save substantial compute cycles
        encoder_output = model.encoder(x=src_tensor, mask=src_mask)

        # Initialize paths: list of tuples -> (cumulative_log_probability, token_ids_list)
        # We start with a single tracking point containing just the <SOS> token ID
        beams = [(0.0, [sos_id])]
        completed_beams = []

        # Autoregressive sequence generation loop
        for step in range(max_len):
            candidates = []

            # Package all current active tracking sequences into a unified parallel batch matrix
            beam_inputs = [tokens for _, tokens in beams]
            dec_input = torch.tensor(beam_inputs, dtype=torch.long, device=device)
            current_beam_size = dec_input.size(0)

            # Replicate encoder states and padding masks to match parallelized beam batching dimensions
            b_encoder_output = encoder_output.repeat(current_beam_size, 1, 1)
            b_src_mask = src_mask.repeat(current_beam_size, 1, 1, 1)

            # Compute Decoder forward pass
            logits = model.decoder(x=dec_input, encoder_output=b_encoder_output, src_mask=b_src_mask)

            # Isolate logits for the absolute LAST predicted token step sequence position
            next_token_logits = logits[:, -1, :]
            # Convert raw outputs into normalized log probabilities: log(P(x))
            log_probs = F.log_softmax(next_token_logits, dim=-1)

            # Snatch top B word choices for each path to minimize sorting load
            topk_log_probs, topk_ids = torch.topk(log_probs, beam_size, dim=-1)

            # Loop through active lanes to expand possible tracks
            for b_idx in range(current_beam_size):
                cum_log_prob, tokens = beams[b_idx]

                for k in range(beam_size):
                    next_token_prob = topk_log_probs[b_idx, k].item()
                    next_token_id = topk_ids[b_idx, k].item()

                    # Accumulate negative log probabilities: log(P(A)) + log(P(B))
                    new_cum_prob = cum_log_prob + next_token_prob
                    new_tokens = tokens + [next_token_id]

                    # Route path to completed container if it outputs an <EOS> boundary token
                    if next_token_id == eos_id:
                        completed_beams.append((new_cum_prob, new_tokens))
                    else:
                        candidates.append((new_cum_prob, new_tokens))

            # Early break check: if we have already gathered enough finished options, stop looping
            if len(completed_beams) >= beam_size:
                break

            # Globally sort all ongoing uncompleted routes and filter down to top B options
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_size]

            if not beams:
                break

        # Safety dump: if no route explicitly produced an <EOS> tag, dump open paths into the pool
        if not completed_beams:
            completed_beams = beams

        # 5. Length Normalization step
        # Shorter sequences accumulate fewer terms and naturally bias negative probabilities higher.
        # We counter this using the length normalization formula: Score = log_prob / (length^alpha)
        normalized_beams = []
        for log_prob, tokens in completed_beams:
            # Exclude the initial starting <SOS> token from length measurements
            length = len(tokens) - 1
            length = max(length, 1)   # Guard against divide-by-zero crashes
            norm_score = log_prob / (length ** alpha)
            normalized_beams.append((norm_score, tokens))

        # Sort and select the highest scoring track candidate
        normalized_beams.sort(key=lambda x: x[0], reverse=True)
        best_tokens = normalized_beams[0][1]

        # 6. Extraction and final text cleanup
        generated_ids = best_tokens[1:]  # Crop out the <SOS> token
        if eos_id in generated_ids:
            generated_ids = generated_ids[:generated_ids.index(eos_id)]  # Slice off <EOS> and everything after

        # 7. Use your Vocabulary.itos helper dictionary to map index positions back to string words
        generated_words = [tgt_vocab.itos.get(idx, "<UNK>") for idx in generated_ids]

        # Merge individual tokens into a final human-readable string sentence
        return " ".join(generated_words)