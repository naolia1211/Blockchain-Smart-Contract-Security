import os
import io
import json
import re
import pyevmasm
from pyevmasm import disassemble_hex
from pyevmasm import assemble_hex

# from solc import compile_source, compile_files, link_code
from solcx import compile_source, compile_files, link_code, get_installed_solc_versions, set_solc_version
from evm_cfg_builder.cfg import CFG
import sys

if __name__ == '__main__':

    file_name = "0.sol"
    compile_version = "0.4.24"
    dot_dir = "test"

    # read the contract source code
    f = open(file_name)
    file_content = f.read()
    f.close()

    result = compile_source(file_content)
    
    # traverse the compiled contracts
    for k,v in result.items():
        k = k.replace('<stdin>:', '')
        print('contract', k)
        cfg = CFG(v['bin-runtime'])
        dot_path = os.path.join(dot_dir, f"{k}.dot")
        digraph_name = k
        all_exit_node = -1

        with open(dot_path, 'w') as file:
            file.write(f'digraph {digraph_name} {{\n')
            file.write('bgcolor=transparent rankdir=UD;\n')
            file.write('node [shape=box style=filled color=black fillcolor=white fontname=arial fontcolor=black];\n')
            
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
                            file.write(f'{all_exit_node} [label="{all_exit_node}: EXIT BLOCK\l" fillcolor=crimson ];')
                        else:
                            file.write(f'{block.start.pc}[label="{ins}" fillcolor=lemonchiffon shape=Msquare ];\n')
                            output_exit.append(block.start.pc)
                            output_exit_instructions.append(ins)
                            end.append(block.start.pc)
                    else:
                        file.write(f'{block.start.pc}[label="{ins}" fillcolor=lemonchiffon shape=Msquare fillcolor=gold ];\n')
                elif not block.all_outgoing_basic_blocks:
                    if block.all_incoming_basic_blocks:
                        if str(block.instructions[-1]) == 'JUMP':
                            jumpt_end.append(block.start.pc)
                            file.write(f'{block.start.pc}[label="{ins}" fillcolor=lemonchiffon ];\n')
                        else:
                            end.append(block.start.pc)
                            file.write(f'{block.start.pc}[label="{ins}" fillcolor=lemonchiffon shape=Msquare color=crimson ];\n')
                else:
                    file.write(f'{block.start.pc}[label="{ins}" fillcolor=lemonchiffon ];\n')

                for son in block.all_outgoing_basic_blocks:
                    node_connections += f'{block.start.pc} -> {son.start.pc};\n'
            
            for i in range(len(jumpt_end)):
                node_connections += f'{jumpt_end[i]} -> {output_exit[i]};\n'
            for end_val in end:
                node_connections += f'{end_val} -> {all_exit_node};\n'
                
            file.write(node_connections)
            file.write('}')
            
        print("{} has been saved successful!".format(dot_path))
        os.system('dot -Tpng {} -o {}'.format(dot_path, os.path.join(dot_dir, '{}.png'.format(k))))