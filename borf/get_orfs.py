# get_orfs.py
from Bio import SeqIO
import numpy as np
import re as re
import pandas as pd
import skbio as skbio
import itertools as itertools

def get_orfs(fasta_file, top_strand = True, min_orf_length = 100, longest_only = True, min_upstream_length = 50):

    all_sequences = []

    # read in fasta file
    for record in SeqIO.parse(fasta_file, 'fasta'):
        all_sequences.append(record)

    # create all frame translations of nt sequence
    aa_frames = []
    frame = []
    seq_length_nt = []
    ids = []
    for seq_string in all_sequences:
        for reading_frame in range(3):

            aa_frames.append(str(skbio.DNA(str(seq_string.seq[reading_frame:])).translate()))
            frame.append(reading_frame)
            seq_length_nt.append(len(str(seq_string.seq)))
            ids.append(seq_string.id)

            if top_strand == False:
                # translate reverse compliment
                aa_frames.append(str(skbio.DNA(str(seq_string.seq[reading_frame:])).complement(reverse=True).translate()))
                frame.append(reading_frame)
                seq_length_nt.append(len(str(seq_string.seq)))
                ids.append(seq_string.id)


    seq_length_nt = np.array(seq_length_nt)
    aa_frames = np.array(aa_frames)
    frame = np.array(frame) + 1
    if top_strand == True:
        strand = np.array([s for s in '+' for i in range(len(aa_frames))])
    else:
        strand = np.tile(np.array(['+','-']) ,len(all_sequences))

    seq_length = np.array([len(o) for o in aa_frames])

    ids = np.array(ids)

    start_sites = []
    stop_sites = []
    orf_sequence = []

    if longest_only == True:

        for aa_seq in aa_frames:

            max_start,max_end = orf_start_stop_from_aa(aa_seq)
            # if returning all > 100AA

            start_sites.append(max_start)
            stop_sites.append(max_end)

            # extract orf sequence
            orf_sequence.append(aa_seq[max_start:max_end])

        orf_sequence = np.array(orf_sequence)

        # check if the last AA is a stop (*) and trim it if neccessary
        last_aa_is_stop = [o[-1] == '*' for o in orf_sequence]
        orf_sequence[last_aa_is_stop] = [o[0:-1] for o in orf_sequence[last_aa_is_stop]]

        orf_length = np.array([len(o) for o in orf_sequence])

        # add 1 to convert pythonic index to normal-person index...
        start_sites = np.array(start_sites) + 1
        stop_sites = np.array(stop_sites)

        # check for upstream ORF?
        # get all sequence upstream of the start (M), and reverse it to find the distance to the nearest upstream stop codon
        first_stop = np.char.find(np.array(aa_frames), "*")
        add_upstream = np.logical_and(np.logical_or(first_stop == -1, first_stop == (stop_sites-1)), start_sites > min_upstream_length)

        if np.any(add_upstream):

            orf_with_upstream = [o[0:s] for o,s in zip(aa_frames[add_upstream], stop_sites[add_upstream])]
            # check if the last AA is a stop (*) and trim it if neccessary
            orf_with_upstream = [replace_last_stop(o) for o in orf_with_upstream]
            orf_sequence[add_upstream] = orf_with_upstream
            start_sites[add_upstream] = 1 #set to 1 for upstream ORFs
            orf_length[add_upstream] = np.array([len(o) for o in orf_sequence[add_upstream]])

        # filter data by minimum orf length
        keep = orf_length > min_orf_length
        aa_frames = aa_frames[keep]
        frame = frame[keep]
        strand = strand[keep]
        seq_length_nt = seq_length_nt[keep]
        ids = np.array(ids)[keep]
        seq_length = seq_length[keep]
        start_sites = start_sites[keep]
        stop_sites = stop_sites[keep]
        orf_sequence = orf_sequence[keep]
        last_aa_is_stop = np.array(last_aa_is_stop)[keep]
        orf_length = orf_length[keep]

        start_site_nt = (start_sites*3) - 3 + frame
        # only give a stop_site_nt location if the last AA is * //// NOT ANYMORE
        stop_site_nt = orf_length*3 + start_site_nt + 3  - frame

        utr3_length = np.zeros(len(start_site_nt))
        utr3_length[last_aa_is_stop]  = seq_length_nt[last_aa_is_stop] - stop_site_nt[last_aa_is_stop]
        utr3_length = utr3_length.astype(int)

        first_aa = [o[0] for o in orf_sequence]
        first_MET = np.where(np.array(first_aa) == 'M', 'M', 'ALT')
        final_stop = np.where(last_aa_is_stop, 'STOP','ALT')

        # collect all and format as pandas DataFrame
        orf_df = pd.DataFrame(index = range(len(start_sites)))
        orf_df['id'] = ids
        orf_df['aa_sequence'] = aa_frames
        orf_df['frame'] = frame
        orf_df['strand'] = strand
        orf_df['seq_length'] = seq_length
        orf_df['seq_length_nt'] = seq_length_nt
        orf_df['orf_sequence'] = orf_sequence
        orf_df['start_site'] = start_sites
        orf_df['stop_site'] = stop_sites
        orf_df['orf_length'] = orf_length
        orf_df['start_site_nt'] = start_site_nt
        orf_df['stop_site_nt'] = stop_site_nt
        orf_df['utr3_length'] = utr3_length
        orf_df['first_MET'] = first_MET
        orf_df['final_stop'] = final_stop

        idx = orf_df.groupby(['id'])['orf_length'].transform(max) == orf_df['orf_length']
        orf_df = orf_df[idx]
        orf_df['isoform_number'] = int(1)

    # if finding all orf > cutoff
    else:
        # make dfs and join////repeat arrays for each orf
        sequence_df = pd.DataFrame(index = range(len(aa_frames)))
        sequence_df['id'] = ids
        sequence_df['aa_sequence'] = aa_frames
        sequence_df['frame'] = frame
        sequence_df['strand'] = strand
        sequence_df['seq_length'] = seq_length
        sequence_df['seq_length_nt'] = seq_length_nt

        sequence_df['seq_index'] = range(len(aa_frames))

        matched_index = []
        isoform_number = []

        for i in range(len(aa_frames)):

            aa_seq = aa_frames[i]
            start_locs,end_locs = orf_start_stop_from_aa(aa_seq, max_only = False)
            # if returning all > 100AA
            orf_lengths = (np.array(end_locs) - np.array(start_locs)) - 1
            above_min_length = orf_lengths >= min_orf_length
            orf_lengths = orf_lengths[above_min_length]
            max_start = np.array(start_locs)[above_min_length]
            max_end = np.array(end_locs)[above_min_length]
            rep_index = np.repeat(i, len(orf_lengths))

            isoform_number.append(np.array(range(1, len(orf_lengths)+1)))
            start_sites.append(max_start)
            stop_sites.append(max_end)
            matched_index.append(rep_index)

            # extract orf sequence
            if np.array(max_start).size == 1:
                orf_sequence.append(aa_seq[int(max_start):int(max_end)])
            elif np.array(max_start).size > 1:
                orf_sequence.append([aa_seq[sta:end] for sta,end in zip(max_start, max_end)])

        start_sites = np.hstack(start_sites)
        stop_sites = np.hstack(stop_sites)
        matched_index = np.hstack(matched_index)
        isoform_number = np.hstack(isoform_number)
        orf_sequence = np.hstack(orf_sequence)

        # check if the last AA is a stop (*) and trim it if neccessary
        last_aa_is_stop = [o[-1] == '*' for o in orf_sequence]
        orf_sequence = np.array([replace_last_stop(o) for o in orf_sequence])

        orf_length = np.array([len(o) for o in orf_sequence])

        # add 1 to convert pythonic index to normal-person index...
        start_sites = np.array(start_sites) + 1
        stop_sites = np.array(stop_sites)

        # check for upstream ORF?
        # get all sequence upstream of the start (M), and reverse it to find the distance to the nearest upstream stop codon
        full_seq_matched = np.array(sequence_df['aa_sequence'][matched_index], dtype='str')

        first_stop = np.char.find(full_seq_matched, "*")
        add_upstream = np.logical_and(np.logical_or(first_stop == -1, first_stop == (stop_sites-1)), start_sites > min_upstream_length)


        if np.any(add_upstream):

            orf_with_upstream = [o[0:s] for o,s in zip(full_seq_matched[add_upstream], stop_sites[add_upstream])]
            # check if the last AA is a stop (*) and trim it if neccessary
            orf_with_upstream = [replace_last_stop(o) for o in orf_with_upstream]
            orf_sequence[add_upstream] = orf_with_upstream
            start_sites[add_upstream] = 1 #set to 1 for upstream ORFs
            orf_length[add_upstream] = np.array([len(o) for o in orf_sequence[add_upstream]])

        # filter data by minimum orf length
        keep = orf_length > min_orf_length

        start_sites = start_sites[keep]
        stop_sites = stop_sites[keep]
        matched_index = matched_index[keep]
        isoform_number = isoform_number[keep]
        orf_sequence = orf_sequence[keep]
        last_aa_is_stop = np.array(last_aa_is_stop)[keep]
        orf_length = orf_length[keep]

        orf_df = pd.DataFrame(index = range(len(orf_sequence)))
        orf_df['seq_index'] = matched_index
        orf_df['orf_sequence'] = orf_sequence
        orf_df['start_site'] = start_sites
        orf_df['stop_site'] = stop_sites
        orf_df['orf_length'] = orf_length
        orf_df['isoform_number'] = isoform_number.astype(int)
        orf_df = pd.merge(sequence_df,orf_df,  on='seq_index', how='right')
        orf_df.drop('seq_index', axis=1,inplace=True)

        orf_df['start_site_nt'] = (orf_df['start_site']*3) - 3 + orf_df['frame']
        # only give a stop_site_nt location if the last AA is * //// NOT ANYMORE
        stop_site_nt = orf_length*3 + orf_df['start_site_nt'] + 3  - orf_df['frame']

        utr3_length = np.zeros(len(stop_site_nt))
        utr3_length[last_aa_is_stop]  = orf_df['seq_length_nt'][last_aa_is_stop] - stop_site_nt[last_aa_is_stop]
        utr3_length = utr3_length.astype(int)

        orf_df['stop_site_nt'] = stop_site_nt
        orf_df['utr3_length'] = utr3_length

        first_aa = [o[0] for o in orf_df['orf_sequence']]
        orf_df['first_MET'] = np.where(np.array(first_aa) == 'M', 'M', 'ALT')
        orf_df['final_stop'] = np.where(last_aa_is_stop, 'STOP','ALT')


    orf_df['orf_class'] = add_orf_classification(orf_df)
    orf_df['fasta_id'] = ('>' + orf_df.id + '.orf' +  orf_df.isoform_number.map(str) +
    ' ' +  orf_df.orf_class + ':' + orf_df.start_site_nt.map(str) +
    '-' + orf_df.stop_site_nt.map(str) + ' strand:' +  orf_df.strand.map(str))

    return orf_df

# return location of the next stop in a sequence, or return the length if no stop found
def find_next_stop(aa_seq, start_loc):
    stop_codon = np.char.find(aa_seq[start_loc:], '*')

    if stop_codon == -1:
        stop_codon = len(aa_seq)-1

    end_loc = stop_codon + start_loc + 1
    return end_loc

def find_max_orf_index(start_locs, end_locs):
    orf_lengths = np.array(end_locs) - np.array(start_locs)
    if orf_lengths.size > 1:
        max_index = np.where(orf_lengths == np.amax(orf_lengths))[0]
        return np.array(start_locs)[max_index][0], np.array(end_locs)[max_index][0]
    else:
        return np.array(start_locs)[0], np.array(end_locs)[0]

def replace_last_stop(orf_seq):

    if orf_seq[-1] == '*':
        replaced_orf_seq = orf_seq[0:-1]
        return replaced_orf_seq
    else:
        return orf_seq



def orf_start_stop_from_aa(aa_seq, max_only = True):
    # find all M
    if aa_seq.count("M") > 0:
        start_locs = []
        end_locs = []

        M_locations = [m.span()[0] for m in re.finditer('M', aa_seq)]
        last_end = -1
        for m in M_locations:
            if m > last_end:
                stop_location = find_next_stop(aa_seq, m)
                start_locs.append(m)
                end_locs.append(stop_location)
                last_end = stop_location
        # if returning all > 100AA
        # find the start/end of the longest ORF
        if max_only == True:
            max_start,max_end = find_max_orf_index(start_locs, end_locs)
        else:
            max_start, max_end = start_locs,end_locs

    else:
        max_start = 0
        max_end = find_next_stop(aa_seq, max_start)

    return max_start,max_end

def add_orf_classification(orf_df):
    orf_class = np.empty(len(orf_df['first_MET']), dtype='object')

    orf_class[np.logical_and(orf_df['first_MET'] == "M", orf_df['final_stop'] == "STOP")] = 'complete' # complete CDS
    orf_class[np.logical_and(orf_df['first_MET'] != "M", orf_df['final_stop'] == "STOP")] = 'incomplete_5prime' # incomplete_5prime
    orf_class[np.logical_and(orf_df['first_MET'] == "M", orf_df['final_stop'] != "STOP")] = 'incomplete_3prime' # incomplete_3prime
    orf_class[np.logical_and(orf_df['first_MET'] != "M", orf_df['final_stop'] != "STOP")] = 'incomplete' # incomplete

    return orf_class

def write_orf_fasta(orf_df, file_out):
    orf_df.to_csv(file_out, index=False, sep='\n', header=False,columns = ['fasta_id','orf_sequence'])

def write_orf_data(orf_df, file_out):
    orf_df = orf_df[['fasta_id', 'id','frame','strand','seq_length_nt', 'start_site_nt', 'stop_site_nt', 'utr3_length', 'start_site', 'stop_site', 'orf_length', 'first_MET', 'final_stop', 'orf_class']]

    orf_df.columns = ['orf_id', 'transcript_id','frame','strand','seq_length_nt', 'start_site_nt', 'stop_site_nt', 'utr3_length_nt', 'start_site_aa', 'stop_site_aa', 'orf_length_aa', 'first_aa_MET', 'final_aa_stop', 'orf_class']

    orf_df.to_csv(file_out, index=False, sep='\t')
