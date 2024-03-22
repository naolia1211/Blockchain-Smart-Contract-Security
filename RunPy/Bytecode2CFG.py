import os
from evm_cfg_builder.cfg import CFG

# Đường dẫn đến thư mục chứa các tệp .bin
input_folder = r"D:\GitHub\Blockchain-Smart-Contract-Security\Dataset\block number dependency (BN)\bytecode"

# Đường dẫn đến thư mục mà bạn muốn lưu các tệp .dot
output_folder = r"D:\GitHub\Blockchain-Smart-Contract-Security\cfg\block number dependency (BN)"

# Duyệt qua tất cả các tệp trong thư mục đầu vào
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith(".bin"):
            # Đọc nội dung tệp .bin
            with open(os.path.join(root, file), 'r') as f:
                bytecode = f.read().strip()

            # Nếu tệp .bin không có nội dung thì bỏ qua
            if not bytecode:
                continue

            # Tạo CFG từ bytecode
            cfg = CFG(bytecode)

            # Tạo thư mục đầu ra nếu nó chưa tồn tại
            output_subfolder = os.path.join(output_folder, os.path.basename(root))
            os.makedirs(output_subfolder, exist_ok=True)

            # Tạo tệp .dot
            dot_file_path = os.path.join(output_subfolder, f"{os.path.splitext(file)[0]}.dot")
            with open(dot_file_path, 'w') as dot_file:
                dot_file.write(f'digraph {os.path.splitext(file)[0]} {{\n')
                dot_file.write('bgcolor=transparent rankdir=UD;\n')
                dot_file.write('node [shape=box style=filled color=black fillcolor=white fontname=arial fontcolor=black];\n')
                node_connections = ''
                output_exit = []
                output_exit_instructions = []
                end = []
                jumpt_end = []
                
                for block in cfg.basic_blocks:
                    ins = ['{}: {}'.format(instruction.pc, str(instruction)) for instruction in block.instructions]
                    ins = '\l'.join(ins) + '\l'
                    
                    if not block.all_incoming_basic_blocks:
                        if not block.all_outgoing_basic_blocks:
                            if str(block.instructions[0]) == 'STOP':
                                all_exit_node = block.instructions[0].pc
                                dot_file.write(f'{all_exit_node} [label="{all_exit_node}: EXIT BLOCK\l" fillcolor=crimson ];')
                            else:
                                dot_file.write(f'{block.start.pc}[label="{ins}" fillcolor=lemonchiffon shape=Msquare ];\n')
                                output_exit.append(block.start.pc)
                                output_exit_instructions.append(ins)
                                end.append(block.start.pc)
                        else:
                            dot_file.write(f'{block.start.pc}[label="{ins}" fillcolor=lemonchiffon shape=Msquare fillcolor=gold ];\n')
                    elif not block.all_outgoing_basic_blocks:
                        if block.all_incoming_basic_blocks:
                            if str(block.instructions[-1]) == 'JUMP':
                                jumpt_end.append(block.start.pc)
                                dot_file.write(f'{block.start.pc}[label="{ins}" fillcolor=lemonchiffon ];\n')
                            else:
                                end.append(block.start.pc)
                                dot_file.write(f'{block.start.pc}[label="{ins}" fillcolor=lemonchiffon shape=Msquare color=crimson ];\n')
                    else:
                        dot_file.write(f'{block.start.pc}[label="{ins}" fillcolor=lemonchiffon ];\n')

                    for son in block.all_outgoing_basic_blocks:
                        node_connections += f'{block.start.pc} -> {son.start.pc};\n'
                
                for i in range(len(jumpt_end)):
                    node_connections += f'{jumpt_end[i]} -> {output_exit[i]};\n'
                for end_val in end:
                    node_connections += f'{end_val} -> {all_exit_node};\n'
                    
                dot_file.write(node_connections)
                dot_file.write('}')
                
            print(f"{dot_file_path} has been saved successfully!")
