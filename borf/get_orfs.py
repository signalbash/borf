# get_orfs.py

import numpy as np
import re as re
import pandas as pd
import skbio as skbio
from Bio import SeqIO
import os


def get_orfs(all_sequences, both_strands=False, min_orf_length=100,
             all_orfs=False, min_upstream_length=50):
    """
    Produce a pandas DataFrame of predicted ORFs from a fasta file.

    Parameters
    ----------
    all_sequences :
        sequence object
    fasta_file : str
        path to the fasta file to predict orfs for
    both_strands : bool
        Provide predictions for both strands? (i.e. reverse compliment).
    min_orf_length : int
        minimum length for a predicted ORF to be reported
    all_orfs : bool
        Return all ORFs longer than min_orf_length?
        Set to False (default) to only return the longest ORF for each sequence.
    min_upstream_length : int
        Minimum length of AA sequence upstream of a canonical start site (e.g. MET) to be used when reporting incomplete_5prime ORFs.
        Upstream sequence starts from the start of the translated sequence, and contains no STOP codons.

    Returns
    -------
    orf_df : DataFrame
        DataFrame containing predicted ORF data and sequences

    """
    # all_sequences = read_fasta(fasta_file)
    # create all frame translations of nt sequence
    ids, aa_frames, frame, strand, seq_length_nt, seq_length = translate_all_frames(all_sequences, both_strands=both_strands)

    if all_orfs is False:

        # find the longest ORF in each frame
        orf_sequence, start_sites, stop_sites, orf_length, last_aa_is_stop = find_longest_orfs(aa_frames)

        # check for upstream ORF?
        # get all sequence upstream of the start (M), and reverse it to find
        # the distance to the nearest upstream stop codon
        orf_sequence, start_sites, orf_length = add_upstream_aas(aa_frames, stop_sites, start_sites, orf_sequence, orf_length, min_upstream_length=min_upstream_length)

        # filter data by minimum orf length
        keep = orf_length >= min_orf_length
        aa_frames, frame, strand, seq_length_nt, ids, seq_length, start_sites, stop_sites, orf_sequence, last_aa_is_stop, orf_length = filter_objects(keep, aa_frames, frame, strand, seq_length_nt, ids, seq_length, start_sites, stop_sites, orf_sequence, last_aa_is_stop, orf_length)

        # only run next steps if there are ORFs
        if np.any(keep):
            # convert aa indices to nt-based indices
            start_site_nt, stop_site_nt, utr3_length = convert_start_stop_to_nt(start_sites, stop_sites, seq_length_nt, orf_length, frame, last_aa_is_stop)

            # check first and last AA
            first_MET = check_first_aa(orf_sequence)
            final_stop = np.where(last_aa_is_stop, 'STOP', 'ALT')
        else:
            start_site_nt = []
            stop_site_nt = []
            utr3_length = []
            first_MET = []
            final_stop = []

        # collect all and format as pandas DataFrame
        orf_df = pd.DataFrame(index=range(len(start_sites)))
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

        # filter by orf with the max length for each sequence
        idx = orf_df.groupby(['id'])['orf_length'].transform(max) == orf_df['orf_length']
        orf_df = orf_df[idx]
        # isoform_number so output format is the same as if all_orfs == True
        orf_df['isoform_number'] = int(1)

    # if finding all orf > cutoff
    else:

        # make DataFrame for each AA frame - joined later with ORF data
        # to prevent increasing the size of this too early
        sequence_df = pd.DataFrame(index=range(len(aa_frames)))
        sequence_df['id'] = ids
        sequence_df['aa_sequence'] = aa_frames
        sequence_df['frame'] = frame
        sequence_df['strand'] = strand
        sequence_df['seq_length'] = seq_length
        sequence_df['seq_length_nt'] = seq_length_nt
        # index so we can match back data later
        sequence_df['seq_index'] = range(len(aa_frames))

        # find all ORFs longer than min_orf_length
        orf_sequence, start_sites, stop_sites, orf_length, last_aa_is_stop, matched_index = find_all_orfs(aa_frames, min_orf_length=min_orf_length)

        # check for upstream ORF?
        # get all sequence upstream of the start (M), and reverse it to
        # find the distance to the nearest upstream stop codon
        full_seq_matched = np.array(sequence_df['aa_sequence'][matched_index], dtype='str')
        orf_sequence, start_sites, orf_length = add_upstream_aas(full_seq_matched, stop_sites, start_sites, orf_sequence, orf_length, min_upstream_length=min_upstream_length)

        # filter data by minimum orf length
        keep = orf_length >= min_orf_length
        start_sites, stop_sites, orf_sequence, last_aa_is_stop, orf_length, matched_index = filter_objects(keep, start_sites, stop_sites, orf_sequence, last_aa_is_stop, orf_length, matched_index)

        # make DataFrame of ORF data
        orf_df = pd.DataFrame(index=range(len(orf_sequence)))
        orf_df['seq_index'] = matched_index
        orf_df['orf_sequence'] = orf_sequence
        orf_df['start_site'] = start_sites
        orf_df['stop_site'] = stop_sites
        orf_df['orf_length'] = orf_length
        # combine with sequence data from above
        orf_df = pd.merge(sequence_df, orf_df,  on='seq_index', how='right')
        orf_df.drop('seq_index', axis=1, inplace=True)

        if np.any(keep):
            # convert aa indices to nt-based indices
            orf_df['start_site_nt'], orf_df['stop_site_nt'], orf_df['utr3_length'] = convert_start_stop_to_nt(start_sites, stop_sites, orf_df['seq_length_nt'], orf_length, orf_df['frame'], last_aa_is_stop)
            # check first and last AA
            orf_df['first_MET'] = check_first_aa(orf_df['orf_sequence'])
            orf_df['final_stop'] = np.where(last_aa_is_stop, 'STOP', 'ALT')
        else:
            # convert aa indices to nt-based indices
            orf_df['start_site_nt'] = []
            orf_df['stop_site_nt'] = []
            orf_df['utr3_length'] = []
            # check first and last AA
            orf_df['first_MET'] = []
            orf_df['final_stop'] = []

        orf_df['isoform_number'] = unique_number_from_list(orf_df.id)

    # add ORF classification
    orf_df['orf_class'] = add_orf_classification(orf_df)
    # Generate ids for writing to fasta
    orf_df['fasta_id'] = ('>' + orf_df.id + '.orf' + orf_df.isoform_number.map(str) + ' ' + orf_df.orf_class + ':' + orf_df.start_site_nt.map(str) + '-' + orf_df.stop_site_nt.map(str) + ' strand:' + orf_df.strand.map(str))

    return orf_df


def translate_all_frames(sequences, both_strands=False):

    """
    translate nt sequences into all 3 frames

    Parameters
    ----------
    sequences : list
        list of nucleotide sequences
    both_strands : bool
        translate both strands?

    Returns
        return ids, aa_seq_by_frame, frame, strand, seq_length_nt, seq_length

    -------
    objects :
        filtered objects
    """
    # create all frame translations of nt sequence
    aa_seq_by_frame = []
    frame = []
    seq_length_nt = []
    ids = []
    for seq_string in sequences:

        nucleotide_seq = str(seq_string.seq)
        non_ATGC = len(nucleotide_seq) - (nucleotide_seq.count('A') + nucleotide_seq.count('T') + nucleotide_seq.count('G') + nucleotide_seq.count('C'))
        skip = non_ATGC > 0

        if skip is False:

            for reading_frame in range(3):

                aa_seq_by_frame.append(str(skbio.DNA(str(seq_string.seq[reading_frame:])).translate()))
                frame.append(reading_frame)
                seq_length_nt.append(len(str(seq_string.seq)))
                ids.append(seq_string.id)

                if both_strands is True:
                    # translate reverse compliment
                    aa_seq_by_frame.append(str(skbio.DNA(str(skbio.DNA(str(seq_string.seq)).complement(reverse=True))[reading_frame:]).translate()))
                    frame.append(reading_frame)
                    seq_length_nt.append(len(str(seq_string.seq)))
                    ids.append(seq_string.id)

        else:
            print("Skipping " + str(seq_string.id) + ". Found " + str(non_ATGC) + " non-ACGT characters.")

    seq_length_nt = np.array(seq_length_nt)
    aa_seq_by_frame = np.array(aa_seq_by_frame)
    frame = np.array(frame) + 1
    if both_strands is False:
        strand = np.array([s for s in '+' for i in range(len(aa_seq_by_frame))])
    else:
        strand = np.tile(np.array(['+', '-']), len(sequences)*3)

    seq_length = np.array([len(o) for o in aa_seq_by_frame])

    ids = np.array(ids)
    return ids, aa_seq_by_frame, frame, strand, seq_length_nt, seq_length


def find_longest_orfs(aa_frames):
    start_sites = []
    stop_sites = []
    orf_sequence = []

    for aa_seq in aa_frames:

        max_start, max_end = orf_start_stop_from_aa(aa_seq)
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
    last_aa_is_stop = np.array(last_aa_is_stop)

    return orf_sequence, start_sites, stop_sites, orf_length, last_aa_is_stop


def orf_start_stop_from_aa(aa_seq, *, max_only=True):
    """
    Find locations of the start (M) and stop (*) codons that produce the
    longest ORF

    Parameters
    ----------
    aa_seq : str
        amino acid sequence
    max_only : bool
        Only return that start and stop locations of the longest ORF

    Returns
    -------
    start_loc : int
        start location
    end_loc : int
        end location

    Examples
    --------

    orf_start_stop_from_aa("META*")
    orf_start_stop_from_aa("META*MEATBORF*")
    orf_start_stop_from_aa("META*MEATBORF")
    orf_start_stop_from_aa("MEATBORF")

    """

    # find all M
    if aa_seq.count("M") > 0:
        start_locs = []
        end_locs = []

        M_locations = [m.span()[0] for m in re.finditer('M', aa_seq)]
        last_end = -1
        for m in M_locations:
            if m > last_end-1:
                stop_location = find_next_stop(aa_seq, m)
                start_locs.append(m)
                end_locs.append(stop_location)
                last_end = stop_location
        # if returning all > 100AA
        # find the start/end of the longest ORF
        if max_only is True:
            max_start, max_end = find_max_orf_index(start_locs, end_locs)
        else:
            max_start, max_end = start_locs, end_locs

    else:
        max_start = 0
        max_end = find_next_stop(aa_seq, max_start)

    return max_start, max_end


def find_next_stop(aa_seq, start_loc):
    """
    Find location of the next stop codon (*) after the start location.
    Return string length if no stop codon is found.

    Parameters
    ----------
    aa_seq : str
        amino acid sequence
    start_loc : int
        start location

    Returns
    -------
    end_loc : int
        location of the next stop codon, or length of string if none is found

    Examples
    --------

    find_next_stop("AAAMBBB*CCC*", 4)
    find_next_stop("AAAMBBBCCC", 4)

    """
    stop_codon = np.char.find(aa_seq[start_loc:], '*')

    if stop_codon == -1:
        stop_codon = len(aa_seq)
        return stop_codon
    else:
        end_loc = stop_codon + start_loc + 1
        return end_loc


def find_max_orf_index(start_locs, end_locs):
    """
    Given sets of start and end locations, return the set with the largest
    difference

    Parameters
    ----------
    start_locs : np.array
        start locations
    end_locs : np.array
        end locations

    Returns
    -------
    start_loc : int
        start location
    end_loc : int
        end location

    Examples
    --------

    find_max_orf_index(start_locs = [0,100], end_locs = [1000, 200])

    """
    orf_lengths = np.array(end_locs) - np.array(start_locs)
    if orf_lengths.size > 1:
        max_index = np.where(orf_lengths == np.amax(orf_lengths))[0]
        return np.array(start_locs)[max_index][0], np.array(end_locs)[max_index][0]
    else:
        return np.array(start_locs)[0], np.array(end_locs)[0]


def add_upstream_aas(aa_frames, stop_sites, start_sites, orf_sequence,
                     orf_length, min_upstream_length=50):
    """
    Add the upstream AAs onto orf sequences

    Parameters
    ----------
    aa_frames : list
        list of translated AA sequences (full length)
    start_sites : list
        list of start sites
    stop_sites : list
        list of stop sites
    orf_sequence : list
        list of ORF sequences (i.e. from start to stop codon)
    orf_length : list
        list of orf lengths
    min_upstream_length : int
        minimum length of upstream sequence for it to be added

    Returns
    -------
    orf_sequence : list
        list of ORF sequences including upstream AA where appropriate
    start_sites : list
        list of start sites
    orf_length : list
        list of orf lengths
    """
    first_stop = np.char.find(np.array(aa_frames), "*")
    add_upstream = np.logical_and(np.logical_or(first_stop == -1, first_stop == (stop_sites-1)), start_sites > min_upstream_length)

    if np.any(add_upstream):
        # object so no sequence truncation
        orf_sequence_withup = orf_sequence.copy().astype('object')
        orf_length_withup = orf_length.copy()
        start_sites_withup = start_sites.copy()

        orf_with_upstream = [o[0:s] for o, s in zip(aa_frames[add_upstream], stop_sites[add_upstream])]
        # check if the last AA is a stop (*) and trim it if neccessary
        orf_with_upstream = [replace_last_stop(o) for o in orf_with_upstream]
        orf_sequence_withup[add_upstream] = orf_with_upstream
        start_sites_withup[add_upstream] = 1  # set to 1 for upstream ORFs
        orf_length_withup[add_upstream] = np.array([len(o) for o in orf_sequence_withup[add_upstream]])

        orf_sequence_withup = orf_sequence_withup.astype(str)

        return orf_sequence_withup, start_sites_withup, orf_length_withup
    else:
        return orf_sequence, start_sites, orf_length


def replace_last_stop(orf_seq):

    """
    replace * with nothing as the final character in in string

    Parameters
    ----------
    orf_seq : str
        orf_sequence

    Returns
    -------
    orf_seq : str
        orf_sequence

    Examples
    --------

    replace_last_stop("META*")
    replace_last_stop("METAL")

    """

    if orf_seq[-1] == '*':
        replaced_orf_seq = orf_seq[0:-1]
        return replaced_orf_seq
    else:
        return orf_seq


def filter_objects(filter, *objects):

    """
    filter multiple objects

    Parameters
    ----------
    filter : list
        boolean list
    objects :
        objects to filter

    Returns
    -------
    objects :
        filtered objects
    """

    new_objects = []
    for o in objects:
        new_objects.append(o[filter])

    return new_objects


def convert_start_stop_to_nt(start_sites, stop_sites, seq_length_nt, orf_length, frame, last_aa_is_stop):
    """
    Convert AA locations to nt locations

    Parameters
    ----------
    start_sites : list
        list of start sites
    stop_sites : list
        list of stop sites
    seq_length_nt : list
        list of sequence lengths (in nt)
    orf_length : list
        list of orf lengths
    frame : list
        list of frames
    last_aa_is_stop : list
        list of bool values for if the stop site refers to the stop codon (*)
        or not.

    Returns
    -------
    start_site_nt : list
        list of start sites (in nt)
    stop_site_nt : list
        list of stop sites (in nt)
    utr3_length : list
        list of 3' utr lengths (in nt)
    """

    start_site_nt = (start_sites*3) - 3 + frame
    # only give a stop_site_nt location if the last AA is * //// NOT ANYMORE
    # using NAN values gives issues when trying to convert to int
    stop_site_nt = orf_length*3 + start_site_nt + 3 - 1
    stop_site_nt[np.logical_not(last_aa_is_stop)] = seq_length_nt[np.logical_not(last_aa_is_stop)]

    utr3_length = np.zeros(len(start_site_nt))
    utr3_length[last_aa_is_stop] = seq_length_nt[last_aa_is_stop] - stop_site_nt[last_aa_is_stop]
    utr3_length = utr3_length.astype(int)
    return start_site_nt, stop_site_nt, utr3_length


def check_first_aa(orf_sequence, start_codon='M'):
    """
    Check that the first AA in a list of ORF sequences is M.

    Parameters
    ----------
    orf_sequence :
        list of orf sequences
    start_codon :
        character representing the start codon

    Returns
    -------
    first_MET : numpy array
        array matching orf_sequence with either the start codon or 'ALT'

    Examples
    --------
    check_first_aa(['META','ETAM'])
    """

    first_aa = [o[0] for o in orf_sequence]
    first_MET = np.where(np.array(first_aa) == start_codon, start_codon, 'ALT')
    return first_MET


def find_all_orfs(aa_frames, min_orf_length):
    matched_index = []
    start_sites = []
    stop_sites = []
    orf_sequence = []

    for i in range(len(aa_frames)):

        aa_seq = aa_frames[i]
        start_locs, end_locs = orf_start_stop_from_aa(aa_seq, max_only=False)
        first_stop = np.char.find(aa_seq, '*')
        # if returning all > 100AA
        # OR if potential upstream incomplete
        orf_lengths = (np.array(end_locs) - np.array(start_locs)) - 1
        above_min_length = np.logical_or(np.logical_or(orf_lengths >= min_orf_length, start_locs < first_stop), first_stop == -1)

        orf_lengths = orf_lengths[above_min_length]
        max_start = np.array(start_locs)[above_min_length]
        max_end = np.array(end_locs)[above_min_length]
        rep_index = np.repeat(i, len(orf_lengths))

        start_sites.append(max_start)
        stop_sites.append(max_end)
        matched_index.append(rep_index)

        # extract orf sequence
        if np.array(max_start).size == 1:
            orf_sequence.append(aa_seq[int(max_start):int(max_end)])
        elif np.array(max_start).size > 1:
            orf_sequence.append([aa_seq[sta:end] for sta, end in zip(max_start, max_end)])

    start_sites = np.hstack(start_sites)
    stop_sites = np.hstack(stop_sites)
    matched_index = np.hstack(matched_index)
    orf_sequence = np.hstack(orf_sequence)

    # check if the last AA is a stop (*) and trim it if neccessary
    last_aa_is_stop = [o[-1] == '*' for o in orf_sequence]
    orf_sequence = np.array([replace_last_stop(o) for o in orf_sequence])

    orf_length = np.array([len(o) for o in orf_sequence])

    # add 1 to convert pythonic index to normal-person index...
    start_sites = np.array(start_sites) + 1
    stop_sites = np.array(stop_sites)
    last_aa_is_stop = np.array(last_aa_is_stop)

    return orf_sequence, start_sites, stop_sites, orf_length, last_aa_is_stop, matched_index


def unique_number_from_list(input_list):
    """
    Produce a list of integers corresponding to the number of times an
    element in the input list has been observed.

    Parameters
    ----------
    input_list : list
        list of values

    Returns
    -------
    occurrence : list
        integer list

    Examples
    --------

    unique_number_from_list(['a','a','b','c','c','c'])
    unique_number_from_list(['a','b','c'])

    """
    dups = {}
    occurrence = []
    for i, val in enumerate(input_list):
        if val not in dups:
            # Store index of first occurrence and occurrence value
            dups[val] = [i, 1]

        # Increment occurrence value,
        else:
            dups[val][1] += 1
            # Use stored occurrence value
        occurrence.append(dups[val][1])
    return occurrence


def add_orf_classification(orf_df):
    """
    Generate ORF type classification from an orf_df.
    complete: Complete CDS - contains start codon and stop codon
    incomplete_5prime: Incomplete CDS - has stop codon, but start of sequence
                       indicates that an upstream start codon may be missing.
    incomplete_3prime: Incomplete CDS - has start codon, but no stop codon.
    incomplete: Incomplete CDS - Both start codon and stop codon not found.

    Parameters
    ----------
    orf_df : DataFrame
        orf_df DataFrame

    Returns
    -------
    orf_class : np.array
        numpy array of orf classifications

    """
    orf_class = np.empty(len(orf_df['first_MET']), dtype='object')

    orf_class[np.logical_and(orf_df['first_MET'] == "M", orf_df['final_stop'] == "STOP")] = 'complete'
    orf_class[np.logical_and(orf_df['first_MET'] != "M", orf_df['final_stop'] == "STOP")] = 'incomplete_5prime'
    orf_class[np.logical_and(orf_df['first_MET'] == "M", orf_df['final_stop'] != "STOP")] = 'incomplete_3prime'
    orf_class[np.logical_and(orf_df['first_MET'] != "M", orf_df['final_stop'] != "STOP")] = 'incomplete'

    return orf_class


def read_fasta(fasta_file):
    """
    read in a fasta file

    Parameters
    ----------
    fasta_file : str
        path to fasta file

    Returns
    -------
    sequences :
        SeqIO records of each sequence
    """
    all_sequences = []

    # read in fasta file
    for record in SeqIO.parse(fasta_file, 'fasta'):
        all_sequences.append(record.upper())

    return all_sequences


def write_orf_data(orf_df, file_out):
    """
    Write ORF sequence metadata to txt file.

    Parameters
    ----------
    orf_df : DataFrame
        orf_df DataFrame
    file_out : str
        path to file to write txt file

    """

    orf_df = orf_df[['fasta_id', 'id', 'frame', 'strand', 'seq_length_nt', 'start_site_nt', 'stop_site_nt', 'utr3_length', 'start_site', 'stop_site', 'orf_length', 'first_MET', 'final_stop', 'orf_class']]

    orf_df.columns = ['orf_id', 'transcript_id', 'frame', 'strand', 'seq_length_nt', 'start_site_nt', 'stop_site_nt', 'utr3_length_nt', 'start_site_aa', 'stop_site_aa', 'orf_length_aa', 'first_aa_MET', 'final_aa_stop', 'orf_class']

    #orf_df.to_csv(file_out, index=False, sep='\t')

    if not os.path.isfile(file_out):
        orf_df.to_csv(file_out, mode='a', index=False, sep='\t')
    elif len(orf_df.columns) != len(pd.read_csv(file_out, nrows=1, sep='\t').columns):
        raise Exception("Columns do not match!! ORF data has " + str(len(orf_df.columns)) + " columns. Output txt file has " + str(len(pd.read_csv(file_out, nrows=1, sep='\t').columns)) + " columns.")
    elif not (orf_df.columns == pd.read_csv(file_out, nrows=1, sep='\t').columns).all():
        raise Exception("Columns and column order of ORF data and txt file do not match!!")
    else:
        orf_df.to_csv(file_out, mode='a', index=False, sep='\t', header=False)


def write_orf_fasta(orf_df, file_out):
    """
    Write ORF sequences to a fasta file.

    Parameters
    ----------
    orf_df : DataFrame
        orf_df DataFrame
    file_out : str
        path to file to write fasta sequences

    """

    orf_df.to_csv(file_out, mode = 'a', index=False, sep='\n', header=False, columns=['fasta_id', 'orf_sequence'])

def batch_iterator(iterator, batch_size):
    """Returns lists of length batch_size.

    This can be used on any iterator, for example to batch up
    SeqRecord objects from Bio.SeqIO.parse(...), or to batch
    Alignment objects from Bio.AlignIO.parse(...), or simply
    lines from a file handle.

    This is a generator function, and it returns lists of the
    entries from the supplied iterator.  Each list will have
    batch_size entries, although the final list may be shorter.
    """
    entry = True  # Make sure we loop once
    while entry:
        batch = []
        while len(batch) < batch_size:
            try:
                entry = next(iterator)
            except StopIteration:
                entry = None
            if entry is None:
                # End of file
                break
            batch.append(entry)
        if batch:
            yield batch
