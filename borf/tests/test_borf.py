import unittest
import pandas as pd
import numpy as np

from borf.get_orfs import read_fasta
from borf.get_orfs import find_next_stop
from borf.get_orfs import find_max_orf_index
from borf.get_orfs import orf_start_stop_from_aa
from borf.get_orfs import find_longest_orfs
from borf.get_orfs import replace_last_stop
from borf.get_orfs import add_upstream_aas
from borf.get_orfs import filter_objects
from borf.get_orfs import translate_all_frames
from borf.get_orfs import convert_start_stop_to_nt
from borf.get_orfs import check_first_aa
from borf.get_orfs import unique_number_from_list
from borf.get_orfs import find_all_orfs
from borf.get_orfs import add_orf_classification
from borf.get_orfs import get_orfs


class TestReadFasta(unittest.TestCase):
    def test_read_fasta(self):

        # check that files are read into correct format"
        read_sequence = read_fasta('test_data/test_mutliple_frame_orfs.fa')
        seq_array = [str(x.seq) for x in read_sequence]
        # check sequence matches
        # (only check first/last few nts, and total length)
        t_start = seq_array[0][0:20] == 'GCTTCGGGTTGGTGTCATGG'
        t_end = seq_array[0][-1:-20:-1] == 'AGTTGTGTTACCGGGACGG'
        t_len = len(seq_array[0]) == 2757

        self.assertTrue(t_start and t_end and t_len)


class TestFindNextStop(unittest.TestCase):

    def test_next_stop_not_longest(self):
        # "check this finds the NEXT stop codon"
        # assert find_next_stop("AAAMBBB*CCC*", 4) == 8
        next_stop = find_next_stop("AMEATBALL*", 0)
        self.assertEqual(next_stop, 10)

    def test_next_stop_from_within(self):
        # "check this finds the NEXT stop codon when given a start position
        # greater than 0/1"
        orf = "AMEATY*METABALL*"
        next_stop = find_next_stop(orf, 7)
        self.assertEqual(next_stop, len(orf))

    def test_next_stop_final(self):
        # "check that this returns the length of the given string when no stop
        # codon is found"
        orf = "AMEATBALL"
        next_stop = find_next_stop(orf, 0)
        self.assertEqual(next_stop, len(orf))


class TestFindMaxOrfIndex(unittest.TestCase):

    def test_find_max_orf_index(self):
        # test basic usage of finding the two maximum values
        self.assertEqual(find_max_orf_index(start_locs=[0, 100],
                                            end_locs=[1000, 200]), (0, 1000))

    def test_find_max_orf_index_offby1(self):
        # test when second index is greater by one
        self.assertEqual(find_max_orf_index(start_locs=[0, 100],
                                            end_locs=[999, 1100]), (100, 1100))

    def test_find_max_orf_index_equal(self):
        # test that first instance of the max is returned
        self.assertEqual(find_max_orf_index(start_locs=[0, 100],
                                            end_locs=[1000, 1100]), (0, 1000))


class TestOrfStartStopFromAA(unittest.TestCase):

    def test_correct_start_stop(self):
        # tests that the correct start/stop locations are given
        # in non-pythonic (1-indexed) manner
        self.assertEqual(orf_start_stop_from_aa('AMEATBALL*'), (1, 10))

    def test_start_stop_no_stop_codon(self):
        # tests that stop location is the final aa when no stop codon is found
        self.assertEqual(orf_start_stop_from_aa('AMEATBALL'), (1, 9))

    def test_start_stop_longest(self):
        # tests that the start/stop locations are given for the LONGEST orf
        self.assertEqual(orf_start_stop_from_aa('MAUL*AMEATBALL'), (6, 14))


class TestFindLongestORF(unittest.TestCase):

    def test_find_longest_orf_output_format(self):
        # tests that a length 5 tupple output, and each is the correct numpy
        # array type
        long_orf = find_longest_orfs(['AMEATBALL'])

        t_len = len(long_orf) == 5
        # test numpy types of all outputs
        t0 = long_orf[0].dtype == '<U8'
        t1 = long_orf[1].dtype == 'int64'
        t2 = long_orf[2].dtype == 'int64'
        t3 = long_orf[3].dtype == 'int64'
        t4 = long_orf[4].dtype == 'bool'

        all_right_types = t0 and t1 and t2 and t3 and t4 and t_len
        self.assertTrue(all_right_types)

    def test_find_longest_orf_trimmed(self):
        # check that the last * is trimmed from the orf sequence
        self.assertEqual(find_longest_orfs(['AMEATBALL*'])[0], ['MEATBALL'])

    def test_find_longest_orf_multiple(self):
        input = ['AMEATBALL*', 'TWOMEATBALLS']
        result = find_longest_orfs(input)
        self.assertEqual(len(result[0]), len(input))

    def test_find_longest_orf_stopsites(self):
        # check that the stop site is calculated as the * for seqs with it,
        # and the last AA for those without
        stop_loc_with_stop = find_longest_orfs(['AMEATBALL*'])[2]
        stop_loc_without_stop = find_longest_orfs(['AMEATBALL'])[2]

        self.assertEqual(stop_loc_with_stop, stop_loc_without_stop + 1)


class TestReplaceLastStop(unittest.TestCase):

    def test_replace_last_stop(self):
        # check that the last * is trimmed from the orf sequence
        self.assertEqual(replace_last_stop('MEATBALL'),
                         replace_last_stop('MEATBALL*'))


class TestAddUpstreamAAs(unittest.TestCase):

    def test_add_upstream_aa_output(self):
        # check all outputs generated and all in correct type
        aa_sequence = np.array(['ALONGERUPSTREAMMEATBALL'])
        orf_sequence, start_sites, stop_sites, orf_length, last_aa_is_stop = find_longest_orfs(
            aa_sequence)
        output = add_upstream_aas(aa_sequence, stop_sites, start_sites,
                                  orf_sequence, orf_length,
                                  min_upstream_length=5)

        t_len = len(output) == 3
        # test numpy types of all outputs
        t0 = output[0].dtype.type == np.str_
        t1 = output[1].dtype == 'int64'
        t2 = output[2].dtype == 'int64'

        all_right_types = t0 and t1 and t2 and t_len
        self.assertTrue(all_right_types)

    def test_add_upstream_aa(self):
        # test expected output
        aa_sequence = np.array(['ALONGERUPSTREAMMEATBALL'])
        orf_sequence, start_sites, stop_sites, orf_length, last_aa_is_stop = find_longest_orfs(
            aa_sequence)
        output = add_upstream_aas(
            aa_sequence,
            stop_sites,
            start_sites,
            orf_sequence,
            orf_length,
            min_upstream_length=5)

        self.assertEqual(output[0], 'ALONGERUPSTREAMMEATBALL')

    def test_add_upstream_aa_multi(self):
        # test with multiple inputs
        aa_sequence = np.array(
            ['ALONGERUPSTREAMMEATBALL', 'TWODOZENMEATBALLS', 'BROWNBEARMAULSGIANTSQUID'])
        orf_sequence, start_sites, stop_sites, orf_length, last_aa_is_stop = find_longest_orfs(
            aa_sequence)
        output = add_upstream_aas(
            aa_sequence,
            stop_sites,
            start_sites,
            orf_sequence,
            orf_length,
            min_upstream_length=5)

        self.assertTrue(np.all(output[0] == np.array(
            ['ALONGERUPSTREAMMEATBALL', 'TWODOZENMEATBALLS', 'BROWNBEARMAULSGIANTSQUID'])))

    def test_add_upstream_aa_noupstream(self):
        # test with no viable upstream AAs
        aa_sequence = np.array(['BEAREATS*MEATBALLS'])
        orf_sequence, start_sites, stop_sites, orf_length, last_aa_is_stop = find_longest_orfs(
            aa_sequence)
        output = add_upstream_aas(
            aa_sequence,
            stop_sites,
            start_sites,
            orf_sequence,
            orf_length,
            min_upstream_length=5)

        self.assertEqual(output[0], 'MEATBALLS')

    def test_add_upstream_aa_shortupstream(self):
        # test with upstream AAs too short
        aa_sequence = np.array(['BEARMEATBALLS'])
        orf_sequence, start_sites, stop_sites, orf_length, last_aa_is_stop = find_longest_orfs(
            aa_sequence)
        output = add_upstream_aas(
            aa_sequence,
            stop_sites,
            start_sites,
            orf_sequence,
            orf_length,
            min_upstream_length=5)

        self.assertEqual(output[0], 'MEATBALLS')

    def test_add_upstream_aa_exactupstream(self):
        # test with upstream AAs of exactly  min_upstream_length
        aa_sequence = np.array(['BEARMEATBALLS'])
        orf_sequence, start_sites, stop_sites, orf_length, last_aa_is_stop = find_longest_orfs(
            aa_sequence)
        output = add_upstream_aas(
            aa_sequence,
            stop_sites,
            start_sites,
            orf_sequence,
            orf_length,
            min_upstream_length=4)

        self.assertEqual(output[0], 'BEARMEATBALLS')


class TestFilterObjects(unittest.TestCase):

    def test_filter_objects(self):
        # check input arrays can be filtered
        letters = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'H', 'I', 'J'])
        values = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
        filter = values < 3
        output = filter_objects(filter, letters, values)

        self.assertTrue(np.all(output[0] == np.array(['A', 'B', 'I', 'J'])) and
                        np.all(output[1] == np.array([1, 2, 2, 1])))


class TestTranslateAllFrames(unittest.TestCase):

    def test_translate_output_format(self):
        # tests that a length 3 tupple output, and each is the correct numpy
        # array type
        sequences = read_fasta('test_data/test_trans_all_frames.fa')
        output = translate_all_frames(sequences, both_strands=False)

        t_len = len(output) == 6
        # test numpy types of all outputs
        t0 = output[0].dtype.type == np.str_
        t1 = output[1].dtype.type == np.str_
        t2 = output[2].dtype == 'int64'
        t3 = output[3].dtype.type == np.str_
        t4 = output[4].dtype == 'int64'
        t5 = output[5].dtype == 'int64'

        all_right_types = t0 and t1 and t2 and t3 and t4 and t5 and t_len
        self.assertTrue(all_right_types)

    def test_translate_allframes(self):
        sequences = read_fasta('test_data/test_trans_all_frames.fa')
        ids, aa_frames, frame, strand, seq_length_nt, seq_length = translate_all_frames(
            sequences, both_strands=False)
        self.assertTrue(np.all(frame == np.array([1, 2, 3])))

    def test_translate_alltransframes(self):
        sequences = read_fasta('test_data/test_trans_all_frames.fa')
        ids, aa_frames, frame, strand, seq_length_nt, seq_length = translate_all_frames(
            sequences, both_strands=False)
        self.assertTrue(np.all(aa_frames == np.array(
            ['MANATEE*', 'WRTRPKN', 'GERDRRI'])))

    def test_translate_posstrand(self):
        sequences = read_fasta('test_data/test_trans_all_frames.fa')
        ids, aa_frames, frame, strand, seq_length_nt, seq_length = translate_all_frames(
            sequences, both_strands=False)
        self.assertTrue(np.all(strand == np.array(['+', '+', '+'])))

    def test_translate_seq_length_nt(self):
        sequences = read_fasta('test_data/test_trans_all_frames.fa')
        ids, aa_frames, frame, strand, seq_length_nt, seq_length = translate_all_frames(
            sequences, both_strands=False)
        self.assertTrue(np.all(seq_length_nt == np.array([24, 24, 24])))

    def test_translate_seq_length(self):
        sequences = read_fasta('test_data/test_trans_all_frames.fa')
        ids, aa_frames, frame, strand, seq_length_nt, seq_length = translate_all_frames(
            sequences, both_strands=False)
        self.assertTrue(np.all(seq_length == np.array([8, 7, 7])))

    def test_translate_bothstrands(self):
        sequences = read_fasta('test_data/test_trans_all_frames.fa')
        ids, aa_frames, frame, strand, seq_length_nt, seq_length = translate_all_frames(
            sequences, both_strands=True)

        frame_correct = np.all(frame == np.array([1, 1, 2, 2, 3, 3]))
        strand_correct = np.all(strand == np.array(
            ['+', '-', '+', '-', '+', '-']))
        trans_correct = np.all(aa_frames == np.array(
            ['MANATEE*', 'LFFGRVRH', 'WRTRPKN', 'YSSVAFA', 'GERDRRI', 'ILRSRSP']))

        self.assertTrue(frame_correct and strand_correct and trans_correct)


class TestConvertAANT(unittest.TestCase):

    def test_convert_nt_output_format(self):
        # tests that a length 3 tupple output, and each is the correct numpy
        # array type
        sequences = read_fasta('test_data/test_frames.fa')
        ids, aa_frames, frame, strand, seq_length_nt, seq_length = translate_all_frames(
            sequences, both_strands=False)
        orf_sequence, start_sites, stop_sites, orf_length, last_aa_is_stop = find_longest_orfs(
            aa_frames)
        # filter data by minimum orf length
        keep = orf_length >= 6
        aa_frames, frame, strand, seq_length_nt, ids, seq_length, start_sites, stop_sites, orf_sequence, last_aa_is_stop, orf_length = filter_objects(
            keep, aa_frames, frame, strand, seq_length_nt, ids, seq_length, start_sites, stop_sites, orf_sequence, last_aa_is_stop, orf_length)

        output = convert_start_stop_to_nt(
            start_sites,
            stop_sites,
            seq_length_nt,
            orf_length,
            frame,
            last_aa_is_stop)

        t_len = len(output) == 3
        # test numpy types of all outputs
        t0 = output[0].dtype == 'int64'
        t1 = output[1].dtype == 'int64'
        t2 = output[2].dtype == 'int64'

        all_right_types = t0 and t1 and t2 and t_len
        self.assertTrue(all_right_types)

    def test_convert_start_nt(self):
        sequences = read_fasta('test_data/test_frames.fa')

        ids, aa_frames, frame, strand, seq_length_nt, seq_length = translate_all_frames(
            sequences, both_strands=False)
        orf_sequence, start_sites, stop_sites, orf_length, last_aa_is_stop = find_longest_orfs(
            aa_frames)
        # filter data by minimum orf length
        keep = orf_length >= 6
        aa_frames, frame, strand, seq_length_nt, ids, seq_length, start_sites, stop_sites, orf_sequence, last_aa_is_stop, orf_length = filter_objects(
            keep, aa_frames, frame, strand, seq_length_nt, ids, seq_length, start_sites, stop_sites, orf_sequence, last_aa_is_stop, orf_length)

        start_site_nt, stop_site_nt, utr3_length = convert_start_stop_to_nt(
            start_sites, stop_sites, seq_length_nt, orf_length, frame, last_aa_is_stop)

        self.assertTrue(np.all(start_site_nt == np.array([1, 2, 3])))

    def test_convert_stop_nt(self):
        sequences = read_fasta('test_data/test_frames.fa')

        ids, aa_frames, frame, strand,seq_length_nt, seq_length = translate_all_frames(sequences, both_strands=False)
        orf_sequence, start_sites, stop_sites, orf_length, last_aa_is_stop = find_longest_orfs(
            aa_frames)
        # filter data by minimum orf length
        keep = orf_length >= 6
        aa_frames, frame, strand, seq_length_nt, ids, seq_length, start_sites, stop_sites, orf_sequence, last_aa_is_stop, orf_length = filter_objects(
            keep, aa_frames, frame, strand, seq_length_nt, ids, seq_length, start_sites, stop_sites, orf_sequence, last_aa_is_stop, orf_length)

        start_site_nt, stop_site_nt, utr3_length = convert_start_stop_to_nt(
            start_sites, stop_sites, seq_length_nt, orf_length, frame, last_aa_is_stop)
        self.assertTrue(np.all(stop_site_nt == np.array([21, 22, 23])))

    def test_convert_stop_nt_3incomplete(self):
        sequences = read_fasta('test_data/test_stopsitent.fa')

        ids, aa_frames, frame, strand,seq_length_nt, seq_length = translate_all_frames(sequences, both_strands=False)
        orf_sequence, start_sites, stop_sites, orf_length, last_aa_is_stop = find_longest_orfs(aa_frames)
        # filter data by minimum orf length
        keep = orf_length >= 6
        aa_frames, frame, strand, seq_length_nt, ids, seq_length, start_sites, stop_sites, orf_sequence, last_aa_is_stop, orf_length = filter_objects(
            keep, aa_frames, frame, strand, seq_length_nt, ids, seq_length, start_sites, stop_sites, orf_sequence, last_aa_is_stop, orf_length)

        start_site_nt, stop_site_nt, utr3_length = convert_start_stop_to_nt(
            start_sites, stop_sites, seq_length_nt, orf_length, frame, last_aa_is_stop)
        self.assertTrue(np.all(stop_site_nt == seq_length_nt))


    def test_convert_utr_nt(self):
        sequences = read_fasta('test_data/test_frames.fa')

        ids, aa_frames, frame, strand, seq_length_nt, seq_length = translate_all_frames(
            sequences, both_strands=False)
        orf_sequence, start_sites, stop_sites, orf_length, last_aa_is_stop = find_longest_orfs(
            aa_frames)
        # filter data by minimum orf length
        keep = orf_length >= 6
        aa_frames, frame, strand, seq_length_nt, ids, seq_length, start_sites, stop_sites, orf_sequence, last_aa_is_stop, orf_length = filter_objects(
            keep, aa_frames, frame, strand, seq_length_nt, ids, seq_length, start_sites, stop_sites, orf_sequence, last_aa_is_stop, orf_length)

        start_site_nt, stop_site_nt, utr3_length = convert_start_stop_to_nt(
            start_sites, stop_sites, seq_length_nt, orf_length, frame, last_aa_is_stop)
        self.assertTrue(np.all(utr3_length == np.array([5, 4, 3])))


class TestCheckFirstAA(unittest.TestCase):

    def test_check_first_aa_pos(self):
        # tests that a length 3 tupple output, and each is the correct numpy
        # array type
        aa_sequence = np.array(['MEATBALL'])
        self.assertEqual(check_first_aa(aa_sequence), 'M')

    def test_check_first_aa_neg(self):
        # tests that a length 3 tupple output, and each is the correct numpy
        # array type
        aa_sequence = np.array(['NOTAMEATBALL'])
        self.assertEqual(check_first_aa(aa_sequence), 'ALT')

    def test_check_first_aa_multi(self):
        # tests that a length 3 tupple output, and each is the correct numpy
        # array type
        aa_sequence = np.array(['MEATBALL', 'NOTAMEATBALL'])
        self.assertTrue(np.all(check_first_aa(
            aa_sequence) == np.array(['M', 'ALT'])))


class TestCheckUniqueN(unittest.TestCase):

    def test_check_unique_n(self):
        # tests that a length 3 tupple output, and each is the correct numpy
        # array type
        values = np.array(
            ['MEATBALL', 'MEATBALL', 'BEAR', 'MEATBALL', 'MEATBALLS'])
        self.assertEqual(unique_number_from_list(values), [1, 2, 1, 3, 1])


class TestFindAllORFs(unittest.TestCase):

    def test_find_all_orfs_output_format(self):

        aa_seqs = np.array(['MEATBALL*MEATBALLBEAR*'])
        output = find_all_orfs(aa_seqs, min_orf_length=5)

        t_len = len(output) == 6
        # test numpy types of all outputs
        t0 = output[0].dtype.type == np.str_
        t1 = output[1].dtype == 'int64'
        t2 = output[2].dtype == 'int64'
        t3 = output[3].dtype == 'int64'
        t4 = output[4].dtype == 'bool'
        t5 = output[5].dtype == 'int64'

        all_right_types = t0 and t1 and t2 and t3 and t4 and t5 and t_len
        self.assertTrue(all_right_types)

    def test_find_two_orfs(self):
        # tests that a length 3 tupple output, and each is the correct numpy
        # array type
        aa_seqs = np.array(['MEATBALL*MEATBALLBEAR*'])
        orf_sequence, start_sites, stop_sites, orf_length, last_aa_is_stop, matched_index = find_all_orfs(
            aa_seqs, min_orf_length=5)

        orf_correct = np.all(orf_sequence == np.array(
            ['MEATBALL', 'MEATBALLBEAR']))
        start_correct = np.all(start_sites == np.array([1, 10]))
        stop_correct = np.all(stop_sites == np.array([9, 22]))
        orf_length_correct = np.all(orf_length == np.array([8, 12]))
        last_aa_is_stop_correct = np.all(
            last_aa_is_stop == np.array([True, True]))
        matched_index_correct = np.all(matched_index == np.array([0, 0]))

        self.assertTrue(
            orf_correct and start_correct and stop_correct and orf_length_correct and last_aa_is_stop_correct and last_aa_is_stop_correct and matched_index_correct)

    def test_find_multi_orfs(self):
        # tests that a length 3 tupple output, and each is the correct numpy
        # array type
        aa_seqs = np.array(['MEATBALL*MEATBALLBEAR*', '*NOPE', 'MELMCAT'])
        orf_sequence, start_sites, stop_sites, orf_length, last_aa_is_stop, matched_index = find_all_orfs(aa_seqs, min_orf_length=5)

        self.assertTrue(np.all(orf_sequence == np.array(['MEATBALL', 'MEATBALLBEAR', 'MELMCAT'])))

    def test_find_multi_orfs_index(self):
        # tests that a length 3 tupple output, and each is the correct numpy
        # array type
        aa_seqs = np.array(['MEATBALL*MEATBALLBEAR*', '*NOPE', 'MELMCAT'])
        orf_sequence, start_sites, stop_sites, orf_length, last_aa_is_stop, matched_index = find_all_orfs(aa_seqs, min_orf_length=5)

        self.assertTrue(np.all(matched_index == np.array([0, 0, 2])))

    def test_find_all_orfs_upstream_ic(self):
        # tests that a length 3 tupple output, and each is the correct numpy
        # array type
        aa_seqs = np.array(['*NOPE', 'YES'])
        orf_sequence, start_sites, stop_sites, orf_length, last_aa_is_stop, matched_index = find_all_orfs(aa_seqs, min_orf_length=5)

        self.assertTrue(np.all(orf_sequence == np.array(['YES'])))

class TestAddOrfClass(unittest.TestCase):

    def test_add_orf_classification_complete(self):
        orf_df = pd.DataFrame(index=range(1))
        orf_df['first_MET'] = 'M'
        orf_df['final_stop'] = 'STOP'

        self.assertTrue(np.all(add_orf_classification(orf_df) ==
                               np.array(['complete'])))

    def test_add_orf_classification_incomplete_5prime(self):
        orf_df = pd.DataFrame(index=range(1))
        orf_df['first_MET'] = 'ALT'
        orf_df['final_stop'] = 'STOP'

        self.assertTrue(np.all(add_orf_classification(orf_df) ==
                               np.array(['incomplete_5prime'])))

    def test_add_orf_classification_incomplete_3prime(self):
        orf_df = pd.DataFrame(index=range(1))
        orf_df['first_MET'] = 'M'
        orf_df['final_stop'] = 'ALT'

        self.assertTrue(np.all(add_orf_classification(orf_df) ==
                               np.array(['incomplete_3prime'])))

    def test_add_orf_classification_incomplete(self):
        orf_df = pd.DataFrame(index=range(1))
        orf_df['first_MET'] = 'ALT'
        orf_df['final_stop'] = 'ALT'

        self.assertTrue(np.all(add_orf_classification(orf_df) ==
                               np.array(['incomplete'])))

    def test_add_orf_classification_multi(self):
        orf_df = pd.DataFrame(index=range(4))
        orf_df['first_MET'] = ['M', 'ALT', 'M', 'ALT']
        orf_df['final_stop'] = ['STOP', 'STOP', 'ALT', 'ALT']

        self.assertTrue(np.all(add_orf_classification(orf_df) ==
                               np.array(['complete', 'incomplete_5prime',
                                         'incomplete_3prime', 'incomplete'])))


class TestGetORFs(unittest.TestCase):

    def test_get_orf_base(self):

        expected = pd.DataFrame(index=range(1))
        expected['id'] = 'Single_FA'
        expected['aa_sequence'] = 'MIMIKL*P'
        expected['frame'] = 1
        expected['strand'] = '+'
        expected['seq_length'] = 8
        expected['seq_length_nt'] = 26
        expected['orf_sequence'] = 'MIMIKL'
        expected['start_site'] = 1
        expected['stop_site'] = 7
        expected['orf_length'] = 6
        expected['start_site_nt'] = 1
        expected['stop_site_nt'] = 21
        expected['utr3_length'] = 5
        expected['first_MET'] = 'M'
        expected['final_stop'] = 'STOP'
        expected['isoform_number'] = 1
        expected['orf_class'] = 'complete'
        expected['fasta_id'] = '>Single_FA.orf1 complete:1-21 strand:+'

        all_sequences = read_fasta('test_data/test_getorfs.fa')
        orf_df = get_orfs(all_sequences, min_orf_length=5)

        self.assertTrue(orf_df.equals(expected))

    def test_get_orf_all(self):

        expected = pd.DataFrame(index=range(2))
        expected['id'] = ['Single_FA', 'Single_FA']
        expected['aa_sequence'] = ['MIMIKL*P', 'GLQLNHDH']
        expected['frame'] = [1, 3]
        expected['strand'] = ['+', '-']
        expected['seq_length'] = [8, 8]
        expected['seq_length_nt'] = [26, 26]
        expected['orf_sequence'] = ['MIMIKL', 'GLQLNHDH']
        expected['start_site'] = [1, 1]
        expected['stop_site'] = [7, 8]
        expected['orf_length'] = [6, 8]
        expected['start_site_nt'] = [1, 3]
        expected['stop_site_nt'] = [21, 26]
        expected['utr3_length'] = [5, 0]
        expected['first_MET'] = ['M', 'ALT']
        expected['final_stop'] = ['STOP', 'ALT']
        expected['isoform_number'] = [1, 2]
        expected['orf_class'] = ['complete', 'incomplete']
        expected['fasta_id'] = ['>Single_FA.orf1 complete:1-21 strand:+',
                                '>Single_FA.orf2 incomplete:3-26 strand:-']

        all_sequences = read_fasta('test_data/test_getorfs.fa')
        orf_df = get_orfs(all_sequences, min_orf_length=5, both_strands=True, all_orfs=True)

        self.assertTrue(orf_df.equals(expected))


if __name__ == '__main__':
    unittest.main()
