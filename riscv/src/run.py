# import args
import os

header_file_list = [x for x in os.listdir() if x.startswith("MMult") and x.endswith(".h")]

def get_header(header):
    return header


def get_gemm_signature(header):
    return f"MY_MMult{header[5:-2]}" 

def gen_main_from_template(header):
    signature = get_gemm_signature(header)
    with open("test_matrix_multiply_template.cpp", "r", encoding="utf-8") as f:
        template = f.read()
    return template.replace("##TO_INCLUDE##", header).replace("##MY_MMult##", signature)

def run(header):
    with open("test_matrix_multiply.cpp", "w", encoding="utf-8") as f:
        f.write(gen_main_from_template(header))
    os.system(f"make run > result_{header}.txt")

def main(run_case=None):
    if run_case:
        run(run_case)
        return
    for header in header_file_list:
        run(header)

if __name__ == '__main__':
    main("MMult_4x4_10.h")