import json
import logging
import numpy as np
import pandas as pd
import re
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt

# Đường dẫn tệp
VOCAB_PATH = 'vocab.json'
TEMP_VOCAB_PATH = 'tempvocab.json'
DEBUG_PATH = 'debug.json'
MAX_VOCAB_SIZE = 50000

# Mapping tên cột
header_mapping = {
    'external_call_function': 'external_call',
    'state_change_after_external_call': 'state_change_post_call',
    'error_handling': 'error_handling',
    'fallback_function_interaction': 'fallback_function',
    'unchecked_external_call': 'unchecked_call',
    'use_of_msg_value_or_sender': 'use_of_msg',
    'delegatecall_with_msg_data': 'delegatecall_with_data',
    'dynamic_delegatecall': 'dynamic_delegatecall',
    'library_safe_practices': 'library_practices',
    'gas_limitations_in_fallback_functions': 'gas_limitations_fallback',
    'balance_check_before_send': 'balance_check_send',
    'multiple_external_calls': 'multiple_calls',
    'external_call_in_loop': 'call_in_loop',
    'reentrancy_guard_missing': 'missing_guard',
    'low_level_call': 'low_level_call',
    'library_usage_in_delegatecall': 'library_delegatecall',
    'state_variables_manipulation': 'state_var_change',
    'state_update_without_verification': 'state_update_no_verification',
    'recursive_calls': 'recursive_calls',
    'context_preservation': 'context_preservation',
    'usage_of_block_timestamp': 'block_timestamp',
    'usage_of_block_number': 'block_number',
    'usage_of_block_blockhash': 'block_hash',
    'miner_manipulation': 'miner_effect',
    'usage_in_comparison_operations': 'comparison_operations',
    'block_number_as_time': 'block_number_time',
    'historical_block_data': 'historical_data',
    'lack_of_private_seed_storage': 'no_seed_storage',
    'predictability_of_randomness_sources': 'predictable_randomness',
    'block_based_random': 'block_random',
    'keccak256_random': 'keccak256_random',
    'unsafe_random_generation': 'random_generation',
    'timestamp_arithmetic': 'timestamp_arithmetic',
    'timestamp_in_condition': 'timestamp_condition',
    'timestamp_assignment': 'timestamp_assignment',
    'timestamp_function_param': 'timestamp_param',
    'block_number_arithmetic': 'block_arithmetic',
    'block_number_in_condition': 'block_condition',
    'block_number_assignment': 'block_assignment',
    'block_number_function_param': 'block_param',
    'block_based_operations': 'block_operations',
    'miner_susceptible_operations': 'miner_operations',
    'input_and_parameter_validation': 'input_validation',
    'owner_variable': 'owner_var',
    'modifier_definition': 'modifier_def',
    'function_visibility': 'function_visibility',
    'authentication_checks': 'auth_checks',
    'state_changing_external_calls': 'state_changing_calls',
    'selfdestruct_usage': 'selfdestruct',
    'vulnerable_sensitive_data_handling': 'sensitive_data',
    'improper_input_padding': 'input_padding',
    'unchecked_address_variables': 'unchecked_address',
    'constructor_ownership_assignment': 'ownership_assignment',
    'transaction_state_variable': 'transaction_state',
    'ownership_transfer': 'ownership_transfer',
    'privileged_operation': 'privileged_op',
    'unchecked_parameter': 'unchecked_param',
    'address_parameter': 'address_param',
    'state_change_after_transfer': 'state_change_transfer',
    'multiple_transfers': 'multiple_transfers',
    'price_check_before_action': 'price_check',
    'unsafe_type_inference': 'unsafe_type',
    'reentrancy_risk': 'reentrancy_risk',
    'unchecked_send_result': 'unchecked_send',
    'assembly_block': 'assembly_block',
    'costly_loop': 'costly_loop',
    'unbounded_operation': 'unbounded_op',
    'gas_intensive_function': 'gas_intensive_func',
    'risky_external_call': 'risky_call',
    'improper_exception_handling': 'exception_handling',
    'arithmetic_in_loop': 'arithmetic_in_loop',
    'risky_fallback': 'risky_fallback',
    'loop_counter_vulnerability': 'loop_counter_issue',
    'gas_limit_in_loop': 'gas_limit_loop',
    'integer_overflow_underflow': 'integer_overflow_underflow',
    'unchecked_arithmetic_vuln': 'unchecked_arithmetic_vuln',
    'division_precision_loss': 'division_precision_loss',
    'increment_decrement_overflow': 'increment_decrement_overflow',
    'unsafe_assignment': 'unsafe_assignment',
    'unsafe_integer_types': 'unsafe_integer_types',
    'unchecked_bounds': 'unchecked_bounds',
    'modulo_bias': 'modulo_bias',
    'multiplication_overflow': 'multiplication_overflow',
    'safemath_missing': 'safemath_missing',
    'floating_point_approximation': 'floating_point_approximation',
    'unsafe_type_casting': 'unsafe_type_casting',
    'unchecked_array_length': 'unchecked_array_length',
    'implicit_type_conversion': 'implicit_type_conversion',
    'arithmetic_in_loop_condition': 'arithmetic_in_loop_condition',
    'division_by_zero': 'division_by_zero',
    'precision_loss_in_division': 'precision_loss_in_division',
    'unsafe_exponentiation': 'unsafe_exponentiation',
    'unchecked_math_function': 'unchecked_math_function',
    'bitwise_operations': 'bitwise_operations',
    'unsafe_math_functions': 'unsafe_math_functions',
    'ternary_arithmetic': 'ternary_arithmetic',
    'arithmetic_operations': 'arithmetic_ops',
    'contract_creation': 'contract_creation',
    'event_logging': 'event_logging',
    'state_variables': 'state_vars',
    'modifiers': 'modifiers',
    'assertions': 'assertions',
    'external_calls': 'external_calls',
    'ether_transfer': 'ether_transfer',
    'fallback_functions': 'fallback_functions',
    'low_level_call_code': 'call_code',
    'dynamic_address_handling': 'dynamic_address',
    'time_based_logic': 'time_logic',
    'complex_state_changes': 'complex_state_changes',
    'unchecked_arithmetic': 'unchecked_arithmetic',
    'unconventional_control_flows': 'unconventional_flows',
    'unchecked_return_values': 'unchecked_return'
}

# Lớp CustomTokenizerWrapper để xử lý token hóa
class CustomTokenizerWrapper:
    def __init__(self, pattern=r'\w+|\{|\}|\(|\)|\[|\]|\;|\=|\+|\-|\*|\/|\!|\%|<|>|\||\&|\.|,|;', vocab_path=VOCAB_PATH, temp_vocab_path=TEMP_VOCAB_PATH, debug_path=DEBUG_PATH):
        self.tokenizer = RegexpTokenizer(pattern)
        self.vocabulary = {'<pad>': 0, '<unk>': 1}
        self.temp_vocabulary = {}
        self.debug_data = []
        self.vocab_path = vocab_path
        self.temp_vocab_path = temp_vocab_path
        self.debug_path = debug_path
        self.max_length = None  # Sẽ được đặt sau dựa trên max_position_embeddings

    def load_vocabulary(self, filename=None):
        filename = filename or self.vocab_path
        try:
            with open(filename, "r") as json_file:
                self.vocabulary = json.load(json_file)
        except FileNotFoundError:
            logging.warning(f"File {filename} not found! Using default vocabulary from pre-trained tokenizer.")

    def save_vocabulary(self, filename=None):
        filename = filename or self.vocab_path
        with open(filename, 'w') as outfile:
            json.dump(self.vocabulary, outfile)

    def save_temp_vocabulary(self, filename=None):
        filename = filename or self.temp_vocab_path
        with open(filename, 'w') as outfile:
            json.dump(self.temp_vocabulary, outfile)

    def save_vocab_with_freq(self, vocab_with_freq, filename=None):
        filename = filename or self.vocab_path.replace('.json', '_with_freq.json')
        with open(filename, 'w') as outfile:
            json.dump(vocab_with_freq, outfile, indent=4)

    def split_camel_case(self, s):
        return re.sub('([a-z])([A-Z])', r'\1 \2', s).split()

    def split_snake_case(self, s):
        return s.split('_')

    def tokenize_column_data(self, column_name, data):
        if not data:
            return ["none"]
        data = str(data)
        tokens = re.split(r'(\W)', data)
        final_tokens = []
        for token in tokens:
            for sub_token in self.split_snake_case(token):
                final_tokens.extend(self.split_camel_case(sub_token))
        return final_tokens

    def convert_token_to_ids(self, tokens):
        ids = []
        for token in tokens:
            if token not in self.vocabulary:
                token = '<unk>'
            ids.append(self.vocabulary[token])
        return ids

    def __call__(self, row):
        combined_data = []
        for col in row.index:
            if col not in ['File Name', 'Vulnerability Label']:
                combined_data.extend(self.tokenize_column_data(col, row[col] if row[col] is not None else ''))
        return combined_data

    def train(self, dataset):
        self.load_vocabulary()
        token_counter = Counter()
        tokenized_data = []
        for _, row in dataset.iterrows():
            tokenized_row = self(row)
            tokenized_data.append(tokenized_row)
            token_counter.update(tokenized_row)
        self.vocab_with_freq = {token: count for token, count in token_counter.items()}
        self.save_vocab_with_freq(self.vocab_with_freq)
        self.temp_vocabulary = self.vocabulary.copy()
        for token, count in token_counter.items():
            if token not in self.temp_vocabulary:
                self.temp_vocabulary[token] = len(self.temp_vocabulary)
        self.save_temp_vocabulary()
        return tokenized_data

def compute_statistics(tokenized_data):#
    lengths = [len(item) for item in tokenized_data]
    logging.info(f"Token lengths: {lengths}")
    max_position_embeddings = int(np.percentile(lengths, 80))  # Đặt giá trị phù hợp với dữ liệu của bạn
    max_position_embeddings = min(max_position_embeddings, 512)
    print(f"max_position_embeddings (length of each padded row): {max_position_embeddings}")
    return max_position_embeddings

def plot_length_distribution(lengths, title, save_path):
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50)
    plt.title(title)
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    plt.savefig(save_path)
    plt.close()

def build_final_vocabulary(temp_vocab_path, vocab_path, vocab_with_freq):
    with open(temp_vocab_path, "r") as temp_file:
        temp_vocab = json.load(temp_file)
    try:
        with open(vocab_path, "r") as vocab_file:
            vocab = json.load(vocab_file)
    except FileNotFoundError:
        vocab = {'<pad>': 0, '<unk>': 1}
    
    sorted_tokens = sorted(vocab_with_freq.items(), key=lambda item: item[1], reverse=True)
    cumulative_count = 0
    freq_threshold = 0
    for token, count in sorted_tokens:
        cumulative_count += 1
        if cumulative_count > MAX_VOCAB_SIZE - len(vocab):
            freq_threshold = count
            break
    
    filtered_tokens = [token for token, count in sorted_tokens if count >= freq_threshold]
    new_tokens = filtered_tokens[:MAX_VOCAB_SIZE - len(vocab)]
    
    # Bắt đầu với <pad> và <unk> có giá trị cố định
    final_vocab = {'<pad>': 0, '<unk>': 1}
    
    # Thêm các token mới vào từ điển, bắt đầu từ chỉ số 2
    current_index = 2
    for token in new_tokens:
        if token not in final_vocab:
            final_vocab[token] = current_index
            current_index += 1
    
    with open(vocab_path, 'w') as vocab_file:
        json.dump(final_vocab, vocab_file)
    
    return final_vocab

def tokenize_dataset(input_csv, output_csv, vocab_path, temp_vocab_path, debug_path):
    tokenizer = CustomTokenizerWrapper(vocab_path=vocab_path, temp_vocab_path=temp_vocab_path, debug_path=debug_path)
    dataset = pd.read_csv(input_csv, dtype=str)
    dataset = dataset.sample(frac=1).reset_index(drop=True)  # Xáo trộn dữ liệu
    # Thay đổi tên cột dựa trên header_mapping
    dataset.rename(columns=header_mapping, inplace=True)
    tokenized_data = tokenizer.train(dataset)
    tokenized_rows = []
    for original_row, tokens in zip(dataset.itertuples(index=False), tokenized_data):
        tokenized_rows.append({
            'File Name': original_row[0],
            'Vulnerability Label': original_row[1],
            'Tokenized_Text': ' '.join(tokens)
        })
    tokenized_df = pd.DataFrame(tokenized_rows)
    tokenized_df.to_csv(output_csv, index=False)
    max_position_embeddings = compute_statistics(tokenized_data)
    plot_length_distribution([len(tokens) for tokens in tokenized_data], 'Token Length Distribution Before Padding and Truncation', 'length_distribution_before.png')
    tokenizer.max_length = max_position_embeddings
    final_vocab = build_final_vocabulary(temp_vocab_path, vocab_path, tokenizer.vocab_with_freq)
    tokenizer.vocabulary = final_vocab
    final_tokenized_data = []
    lengths_after_padding = []
    for tokens in tokenized_data:
        if len(tokens) > tokenizer.max_length:
            tokens = tokens[:tokenizer.max_length]
        elif len(tokens) < tokenizer.max_length:
            tokens.extend(['<pad>'] * (tokenizer.max_length - len(tokens)))
        input_ids = tokenizer.convert_token_to_ids(tokens)
        attention_mask = [1 if token != '<pad>' else 0 for token in tokens]
        final_tokenized_data.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'tokens': tokens
        })
        lengths_after_padding.append(len(tokens))
    plot_length_distribution(lengths_after_padding, 'Token Length Distribution After Padding and Truncation', 'length_distribution_after.png')
    final_tokenized_rows = []
    for original_row, item in zip(dataset.itertuples(index=False), final_tokenized_data):
        final_tokenized_rows.append({
            'File Name': original_row[0],
            'Vulnerability Label': original_row[1],
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'Tokenized_Text': ' '.join(item['tokens'])
        })
    final_tokenized_df = pd.DataFrame(final_tokenized_rows)
    final_tokenized_df.to_csv(output_csv.replace('.csv', '_final.csv'), index=False)
    return max_position_embeddings, final_tokenized_data, final_tokenized_df, final_vocab

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tokenize Solidity dataset for machine learning.")
    parser.add_argument('--input_csv', type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument('--output_csv', type=str, required=True, help="Path to the output CSV file.")
    parser.add_argument('--vocab_path', type=str, default=VOCAB_PATH, help="Path to the vocabulary JSON file.")
    parser.add_argument('--temp_vocab_path', type=str, default=TEMP_VOCAB_PATH, help="Path to the temporary vocabulary JSON file.")
    parser.add_argument('--debug_path', type=str, default=DEBUG_PATH, help="Path to the debug JSON file.")

    args = parser.parse_args()

    max_position_embeddings, tokenized_data, final_tokenized_df, final_vocab = tokenize_dataset(
        args.input_csv, args.output_csv, args.vocab_path, args.temp_vocab_path, args.debug_path
    )
    print(f"Recommended max_position_embeddings: {max_position_embeddings}")
