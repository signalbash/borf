
'''borf.borf: for running borf'''


import argparse
import os
import sys
import re
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
    parser.add_argument('-c', '--genetic_code', type=int, default=1, help='Genetic code (int: 1-14) to use for translation (default: %(default)d). See https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi for list')
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

    strand_warning = False
    for i, batch in enumerate(batch_iterator(record_iter, batch_size)):
        all_sequences = []
        for record in batch:
            all_sequences.append(record.upper())


        if i == 0:
            # check strandedness

            orf_data = get_orfs(all_sequences, both_strands=True,
                                min_orf_length=args.orf_length, all_orfs=True,
                                min_upstream_length=args.upstream_incomplete_length,
                                genetic_code=args.genetic_code)

            orf_data_strand_bias = orf_data.sort_values(by='orf_length', ascending = False)
            orf_data_strand_bias = orf_data_strand_bias.drop_duplicates('id', keep='first')

            if len(orf_data_strand_bias) >= 10:
                pos_neg_bias = orf_data_strand_bias['strand'][orf_data_strand_bias['orf_class'] == "complete"].value_counts()
                positive_strand_bias = pos_neg_bias[0] / (pos_neg_bias[0]+pos_neg_bias[1])
                if positive_strand_bias > 0.7 and args.strand == True:
                    #data is likely from a stranded assembly.
                    print("Are you sure your input .fasta file isn't stranded?")
                    print(str(positive_strand_bias*100)+ "% of transcripts have the longest ORF on the + strand")
                    strand_warning = True

                if positive_strand_bias <= 0.7 and args.strand == False:
                    print("Are you sure your input .fasta file is stranded?")
                    print(str(positive_strand_bias*100)+ "% of transcripts have the longest ORF on the + strand")
                    strand_warning = True

            if args.strand == False:
                orf_data = orf_data[orf_data['strand'] == '+']
            if args.all_orfs == False:
                idx = orf_data.groupby(['id'])['orf_length'].transform(max) == orf_data['orf_length']
                orf_data = orf_data[idx]
                orf_data['isoform_number'] = 1
                orf_data['fasta_id'] = [re.sub("[.]orf[0-9]*",".orf1", x) for x in orf_data['fasta_id']]

        else:
            orf_data = get_orfs(all_sequences, both_strands=args.strand,
                                min_orf_length=args.orf_length, all_orfs=args.all_orfs,
                                min_upstream_length=args.upstream_incomplete_length,
                                genetic_code=args.genetic_code)

        write_orf_fasta(orf_data, output_path_pep)
        write_orf_data(orf_data, output_path_txt)

        start_seq_n = (i*batch_size) + 1
        end_seq_n = min(start_seq_n + (batch_size - 1), n_seqs)
        print("Processed sequences " + str(start_seq_n) + " to " + str(end_seq_n) + " of " + str(n_seqs))

    print("Done with borf.")
    print("Results in " + output_path_pep + " and " + output_path_txt)

    if strand_warning == True:
        print("This data caused a warning based on strandedness. Please check the top of the log for details and rerun with appropriate flags if neccessary.")
