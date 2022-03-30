import os
import NeuroAnalysisTools.MotionCorrection as mc
import h5py
from multiprocessing import Pool

def run():
    data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data_rabies_project" \
                  r"\180816-M376019-zstack\zstack_2p_zoom2"
    ref_ch_n = 'red'
    apply_ch_ns = ['red']
    n_process = 8

    step_ns = [f for f in os.listdir(os.path.join(data_folder, ref_ch_n))]
    step_ns = [f for f in step_ns if os.path.isdir(os.path.join(data_folder, ref_ch_n, f))]
    step_ns.sort()

    chunk_p = Pool(n_process)

    for ch_n in apply_ch_ns:
        mc_params = []
        for step_n in step_ns:
            movie_path = os.path.join(data_folder, ch_n, step_n, step_n + '.tif')
            offsets_path = os.path.join(data_folder, ref_ch_n, step_n, 'correction_offsets.hdf5')
            mc_params.append((movie_path, offsets_path))

        chunk_p.map(apply_offset_single, mc_params)


def apply_offset_single(param):

    movie_path, offsets_path = param

    offsets_f = h5py.File(offsets_path)
    ref_path = offsets_f['file_0000'].attrs['path']
    offsets_f.close()

    mc.apply_correction_offsets(offsets_path=offsets_path,
                                path_pairs=[[ref_path, movie_path]],
                                output_folder=os.path.split(os.path.realpath(movie_path))[0],
                                process_num=1,
                                fill_value=0.,
                                avi_downsample_rate=None,
                                is_equalizing_histogram=False)

if __name__ == '__main__':
    run()