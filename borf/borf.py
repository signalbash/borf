
import argparse
import os
from get_orfs import read_fasta,get_orfs,write_orf_fasta,write_orf_data

parser = argparse.ArgumentParser(description='Get orf predicitions from a nucleotide fasta file')

parser.add_argument('Fasta', metavar='fasta_file', type=str, help='fasta file to predict ORFs')
parser.add_argument('-o', '--output_path', type = str, help='path to write output files. [OUTPUT_PATH].pep and [OUTPUT_PATH].txt')

parser.add_argument('-s', '--strand', action='store_true', help='Predict orfs for both strands')
parser.add_argument('-a', '--all_orfs', action='store_true', help='Return all ORFs for each sequence longer than the cutoff')
parser.add_argument('-l', '--orf_length',type=int, default = 100, help='Minimum ORF length (AA)')
parser.add_argument('-u', '--upstream_incomplete_length',type=int, default = 50, help='Minimum length (AA) of uninterupted sequence upstream of ORF to be included for incomplete_5prime transcripts')


args = parser.parse_args()

input_file = args.Fasta

if args.output_path is None:

    output_path = os.path.splitext(input_file)[0]
else:
    output_path = args.output_path

output_path_pep = output_path + '.pep'
output_path_txt = output_path + '.txt'


orf_data = get_orfs(input_file, both_strands = args.strand, min_orf_length = args.orf_length, all_orfs = args.all_orfs, min_upstream_length = args.upstream_incomplete_length)

write_orf_fasta(orf_data, output_path_pep)
write_orf_data(orf_data, output_path_txt)
