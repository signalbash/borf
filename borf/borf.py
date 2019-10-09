
'''borf.borf: for running borf'''


import argparse
import os
import sys
from Bio import SeqIO
from .get_orfs import get_orfs, write_orf_fasta, write_orf_data, batch_iterator


def main():

    parser = argparse.ArgumentParser(description='Get orf predicitions from a nucleotide fasta file')

    parser.add_argument('Fasta', metavar='fasta_file', type=str, help='fasta file to predict ORFs')
    parser.add_argument('-o', '--output_path', type=str, help='path to write output files. [OUTPUT_PATH].pep and [OUTPUT_PATH].txt (default: input .fa file name)')
    parser.add_argument('-s', '--strand', action='store_true', help='Predict orfs for both strands')
    parser.add_argument('-a', '--all_orfs', action='store_true', help='Return all ORFs for each sequence longer than the cutoff')
    parser.add_argument('-l', '--orf_length', type=int, default=100, help='Minimum ORF length (AA). (default: %(default)d)')
    parser.add_argument('-u', '--upstream_incomplete_length', type=int, default=50, help='Minimum length (AA) of uninterupted sequence upstream of ORF to be included for incomplete_5prime transcripts (default: %(default)d)')
    parser.add_argument('-b', '--batch_size', type=int, default=10000, help='Number of fasta records to read in in each batch')
    parser.add_argument('-f', '--force_overwrite', action='store_true', help='Force overwriting of output files?')

    args = parser.parse_args()

    input_file = args.Fasta

    if args.output_path is None:
        output_path = os.path.splitext(input_file)[0]
    else:
        output_path = args.output_path

    output_path_pep = output_path + '.pep'
    output_path_txt = output_path + '.txt'

    # check if files exist already
    if os.path.isfile(output_path_pep) or os.path.isfile(output_path_txt):

        if os.path.isfile(output_path_pep) and os.path.isfile(output_path_txt):
             print(output_path_pep + " and " + output_path_txt + " already exist")
        elif os.path.isfile(output_path_pep):
            print(output_path_pep + " already exists")
        else:
            print(output_path_txt + " already exists")

        if not args.force_overwrite:
            overwrite = input("Do you want to overwrite these files? ([Y]/n): ").lower().strip()[:1]
            if not (overwrite == "y" or overwrite == ""):
                sys.exit(1)
            else:
                # remove old files so you don't append new data to old files
                if os.path.isfile(output_path_pep):
                    os.remove(output_path_pep)
                if os.path.isfile(output_path_txt):
                    os.remove(output_path_txt)
        else:
            print('Overwriting files')
            if os.path.isfile(output_path_pep):
                os.remove(output_path_pep)
            if os.path.isfile(output_path_txt):
                os.remove(output_path_txt)

    # number of sequences
    n_seqs = 0
    for record in SeqIO.parse(input_file, 'fasta'):
        n_seqs += 1

    batch_size = args.batch_size

    record_iter = SeqIO.parse(open(input_file), 'fasta')

    for i, batch in enumerate(batch_iterator(record_iter, batch_size)):
        all_sequences = []
        for record in batch:
            all_sequences.append(record.upper())
        orf_data = get_orfs(all_sequences, both_strands=args.strand,
                            min_orf_length=args.orf_length, all_orfs=args.all_orfs,
                            min_upstream_length=args.upstream_incomplete_length)

        write_orf_fasta(orf_data, output_path_pep)
        write_orf_data(orf_data, output_path_txt)

        start_seq_n = (i*batch_size) + 1
        end_seq_n = min(start_seq_n + (batch_size - 1), n_seqs)
        print("Processed sequences " + str(start_seq_n) + " to " + str(end_seq_n) + " of " + str(n_seqs))

    print("Done with borf.")
    print("Results in " + output_path_pep + " and " + output_path_txt)
