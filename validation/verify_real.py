"""Independent output checks and GPU energy recomputation for the saved real runs."""
import argparse
import json
import os
from pathlib import Path
import queue
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import crest_mace as cm
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true',
                        help='Check archived outputs without loading a model or GPU')
    parser.add_argument('--model', type=Path, default=ROOT.parent / 'models/MACE-omol.model',
                        help='Local OMol model for independent energy recomputation')
    args = parser.parse_args()
    report = {}
    reference = cm.read_xyz_frames(ROOT.parent / 'input/iso_321.xyz')[0]
    ref_pos = np.array(reference[1])
    for name in ['final_sp', 'final_opt', 'final_stage1']:
        work = ROOT / name / 'iso_321'
        frames = cm.read_xyz_frames(work / 'crest_ensemble.xyz')
        assert frames
        for symbols, pos in frames:
            assert symbols == reference[0]
            np.testing.assert_allclose(np.array(pos)[[0, 1, 2, 4]], ref_pos[[0, 1, 2, 4]], atol=1e-8, rtol=0)
        lines = (work / 'crest_ensemble.xyz').read_text().splitlines()
        energies = [float(lines[i + 1]) for i in range(0, len(lines), 134)]
        assert energies == sorted(energies)
        assert cm.read_lowest_energy(work / 'crest_best.xyz') == energies[0]
        assert cm.read_xyz_frames(work / 'crest_best.xyz') == frames[:1]
        if name != 'final_stage1':
            assert (energies[-1] - energies[0]) * cm.HARTREE_TO_KCAL <= 6.00001
            ts = cm.read_xyz_frames(work / 'ts_ensemble.xyz')
            np.testing.assert_array_equal(ts[0][1], reference[1])
            assert ts[1:] == frames
            assert (work / 'initial_true_ts.xyz').read_bytes() == (ROOT.parent / 'input/iso_321.xyz').read_bytes()
        manifest = json.loads((work.parent / 'crest_mlip_manifest.json').read_text())
        assert manifest['script_sha256'] == cm.file_sha256(cm.__file__)
        assert manifest['inputs_sha256']['iso_321.xyz'] == cm.file_sha256(ROOT.parent / 'input/iso_321.xyz')
        assert '1 跳过' in (ROOT / (name + '.resume.log')).read_text()
        report[name] = dict(n_raw=cm.count_conformers(work / 'crest_conformers_gfnff.xyz'),
                            n_final=len(frames), lowest_Eh=energies[0],
                            freeze=json.loads((work / 'freeze_validation.json').read_text()))
        if (work / 'optimization_status.json').exists():
            report[name]['optimization'] = json.loads((work / 'optimization_status.json').read_text())

    if args.check_only:
        print('PASS: archived SP, opt and Stage1 outputs, frozen coordinates, '
              'energy ordering, window, original input, manifests and resume logs')
        return
    if not args.model.is_file():
        parser.error(f'Model not found: {args.model}; use --check-only to check outputs without a model')

    # Two distinct inputs guarantee both GPU workers receive a task, even if
    # the short CREST search happened to produce only one conformer.
    cfg = cm.load_config(ROOT.parent / 'examples/config_iso321_sp.yaml')['mlip']
    cfg['model'] = str(args.model.resolve())
    cfg['work'] = str(ROOT / 'two_gpu')
    moved = ref_pos.copy()
    moved[10, 0] += 0.02
    frames = [reference, (reference[0], moved.tolist())]
    pools, rq = cm.init_gpu_pools([0, 1], cfg['model'], False, 0, 1)
    try:
        results = cm.run_mlip_stage2(frames, cfg, pools, rq, threading.Lock(), queue.Queue(), 1, 'two_gpu')
        assert all(result[0] == 'ok' for result in results)
    finally:
        for pool in pools:
            pool.terminate()
            pool.join()

    # Fresh calculator through the public ASE API; do not reuse workflow results.
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from ase import Atoms
    from mace.calculators import MACECalculator
    calc = MACECalculator(model_paths=cfg['model'], device='cuda:0', enable_cueq=False)
    def energy(frame):
        atoms = Atoms(frame[0], positions=frame[1])
        atoms.info.update(charge=0, spin=1)
        atoms.calc = calc
        return float(atoms.get_potential_energy())
    report['two_gpu_independent_errors_eV'] = []
    for frame, result in zip(frames, results):
        delta = abs(energy(frame) - result[1])
        assert delta < 1e-6, delta
        report['two_gpu_independent_errors_eV'].append(delta)
    report['recomputed_output_errors_eV'] = {}
    for name in ['final_sp', 'final_opt']:
        work = ROOT / name / 'iso_321'
        best = cm.read_xyz_frames(work / 'crest_best.xyz')[0]
        delta = abs(energy(best) - cm.read_lowest_energy(work / 'crest_best.xyz') / cm.EV_TO_HARTREE)
        assert delta < 1e-5, delta  # XYZ rounds coordinates and energy to 8 decimals.
        report['recomputed_output_errors_eV'][name] = delta
        original_sp = float((work / 'ts_ensemble.xyz').read_text().splitlines()[1].split()[0])
        assert abs(energy(reference) - original_sp / cm.EV_TO_HARTREE) < 1e-6
    (ROOT / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
