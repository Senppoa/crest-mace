"""Regression tests: python -m unittest -v test_crest_mace (ASE, NumPy, PyYAML)."""
import contextlib
import io
import json
import queue
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml
import crest_mace as cm

ROOT = Path(__file__).resolve().parent


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.xyz = self.root / 'iso_321.xyz'
        shutil.copyfile(ROOT / 'input/iso_321.xyz', self.xyz)
        self.syms, self.pos = cm.read_xyz_frames(self.xyz)[0]
        self.cfg = dict(input=str(self.root), output=str(self.root / 'out'),
                        stage1_only=True, resume=True, freeze=[1, 2, 3, 5],
                        hardfreeze=[1, 2, 3, 5], chrg=0)

    def load(self, **changes):
        path = self.root / 'config.yaml'
        path.write_text(yaml.safe_dump(dict(self.cfg, **changes)))
        return cm.load_config(path)

    def pipeline(self, mlip=None, frames=None, hard=None, resume=True):
        cfg = self.load()
        m = dict(cfg['mlip'], max_conformers=None)
        m.update(mlip or {})
        if frames is None:
            frames = [(-10.0, self.syms, (np.array(self.pos) + [4, -2, 1]).tolist()),
                      (-11.0, self.syms, self.pos)]
        def command(cmd, cwd, log, env, full=False):
            if '--constrain' in cmd:
                (cwd / '.xcontrol.sample').write_text('$constrain\n force constant=0.5\n$end\n')
                (cwd / 'coord.ref').write_text('reference')
            else:
                cm.write_ensemble_xyz(cwd / 'crest_conformers.xyz', frames)
            return 0
        cm._stage1_slots = threading.BoundedSemaphore(1)
        self.events = queue.Queue()
        with patch.object(cm, 'run_cmd', side_effect=command) as run:
            result = cm.process_structure(1, self.xyz, self.root / 'out', 1, 0, 0,
                [1, 2, 3, 5], 0, resume, False, False, True, m, [], None,
                threading.Lock(), None, self.events, {}, threading.Lock(),
                hard_freeze=hard or [1, 2, 3, 5], fc_pos=10, crest_mdlen='0.1')
            self.command_calls = run.call_count
        return result

    def test_input_roundtrip(self):
        self.assertEqual(len(self.syms), 132)
        self.assertEqual(self.syms[0], 'Rh')
        target = self.root / 'frames.xyz'
        cm.write_ensemble_xyz(target, [(-10, self.syms, self.pos)] * 2)
        self.assertEqual(cm.count_conformers(target), 2)
        self.assertEqual(cm.read_xyz_frames(target), [(self.syms, self.pos)] * 2)

    def test_invalid_xyz(self):
        for text in ['1\ncomment\n', '0\n\n', 'x\n\n',
                     '1\n\nC nan 0 0\n', '1\n\nC 0 0\n', '1\n\nXx 0 0 0\n']:
            with self.subTest(text=text):
                self.xyz.write_text(text)
                with self.assertRaises(ValueError):
                    cm.read_xyz_frames(self.xyz)

    def test_config_invalid_types(self):
        for changes in [dict(threads=True), dict(workers=0), dict(stage1_workers=2),
                        dict(freeze=[1, 1]), dict(freeze=[0]), dict(hardfreeze=[133, 133]),
                        dict(resume='true'), dict(fc_pos=0), dict(hardfreeze_tol_A=-1),
                        dict(unknown=True), dict(mlip=[]), dict(mlip=False),
                        dict(mlip={'ewin_kcal_mol':-1}), dict(crest_mdlen='x0')]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.load(**changes)

    def test_config_normalization(self):
        cfg = self.load(freeze='1,2,3,5', crest_mdlen='X 2.0', uhf=2,
                        freeze_by_structure={'iso_321':[1, 3]})
        self.assertEqual(cfg['mlip']['spin'], 3)
        self.assertEqual(cfg['crest_mdlen'], 'x2.0')
        self.assertEqual(cm.structure_atom_lists(cfg, 'iso_321')['hardfreeze'], [1, 3])
        self.assertEqual(cfg['mlip']['freeze_0based'], [0, 1, 2, 4])

    def test_manifest_stage1_ignores_unused_model(self):
        cfg = self.load(mlip={'model':'missing.model'})
        out = self.root / 'out'
        cm.prepare_output(out, cfg, [self.xyz])
        cm.prepare_output(out, cfg, [self.xyz])
        self.xyz.write_text(self.xyz.read_text() + '\n')
        with self.assertRaises(ValueError):
            cm.prepare_output(out, cfg, [self.xyz])

    def test_stage1_sort_freeze_and_resume(self):
        result = self.pipeline()
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['lowest_energy'], -11)
        work = Path(result['work'])
        frames = cm.read_xyz_frames(work / 'crest_ensemble.xyz')
        for _, pos in frames:
            np.testing.assert_allclose(np.array(pos)[[0, 1, 2, 4]], np.array(self.pos)[[0, 1, 2, 4]], atol=1e-8)
        validation = json.loads((work / 'freeze_validation.json').read_text())
        self.assertAlmostEqual(validation['max_rigid_shift_A'], np.sqrt(21), places=7)
        self.assertIn('force constant=10', (work / 'constraints.inp').read_text())
        self.assertIn('freeze = "1,2,3,5"', (work / 'sampling.toml').read_text())
        result = self.pipeline(mlip={'include_initial_ts':True})
        self.assertEqual(result['status'], 'skipped')
        self.assertEqual(self.command_calls, 0)

    def test_rigid_rotation_preserves_geometry(self):
        rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        result = self.pipeline(frames=[(-10, self.syms, (np.array(self.pos) @ rotation + 3).tolist())])
        _, out_pos = cm.read_xyz_frames(Path(result['work']) / 'crest_best.xyz')[0]
        np.testing.assert_allclose(out_pos, self.pos, atol=1e-7)

    def test_deformed_hardfreeze_rejected(self):
        pos = np.array(self.pos)
        pos[1] += 2
        result = self.pipeline(frames=[(-12, self.syms, pos.tolist()), (-10, self.syms, self.pos)])
        self.assertEqual(result['n_final'], 1)
        self.assertEqual(result['lowest_energy'], -10)

    def test_nonfinite_gfnff_rejected(self):
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            self.pipeline(frames=[(float('nan'), self.syms, self.pos)])

    def test_actual_hardfreeze_list_and_limit_after_rejection(self):
        pos = np.array(self.pos)
        pos[5] += 2
        result = self.pipeline(mlip={'max_conformers':1}, hard=[1, 2, 3, 5, 6],
            frames=[(-12, self.syms, pos.tolist()), (-10, self.syms, self.pos),
                    (-9, self.syms, self.pos)])
        self.assertEqual(result['n_final'], 1)
        self.assertEqual(result['lowest_energy'], -10)
        report = json.loads((Path(result['work']) / 'freeze_validation.json').read_text())
        self.assertEqual(report['selected_gfnff_ranks'], [2])

    def test_no_eligible_conformers(self):
        pos = np.array(self.pos)
        pos[1] += 2
        with self.assertRaisesRegex(ValueError, 'No conformers pass'):
            self.pipeline(frames=[(-10, self.syms, pos.tolist())])

    def test_atom_order_mismatch(self):
        syms = list(self.syms)
        syms[0] = 'C'
        with self.assertRaisesRegex(ValueError, 'atom order differs'):
            self.pipeline(frames=[(-10, syms, self.pos)])

    def test_stage2_sort_window_initial_and_partial_failure(self):
        # The original input may legally lack a final newline.
        self.xyz.write_text(self.xyz.read_text().rstrip('\n'))
        def stage2(frames, cfg, *args):
            cfg['initial_result'] = ('ok', -200, self.pos, 0, True)
            return [('ok', -100, self.pos, 0, True), ('ok', -101, self.pos, 0, True)]
        with patch.object(cm, 'run_mlip_stage2', side_effect=stage2):
            result = self.pipeline(mlip={'stage1_only':False, 'include_initial_ts':True})
        self.assertEqual(result['n_final'], 1)
        self.assertAlmostEqual(result['lowest_energy'], -101 * cm.EV_TO_HARTREE)
        work = Path(result['work'])
        self.assertEqual(cm.count_conformers(work / 'ts_ensemble.xyz'), 2)
        self.assertEqual(len(cm.read_xyz_frames(work / 'ts_ensemble.xyz')), 2)
        self.assertEqual((work / 'initial_true_ts.xyz').read_bytes(), self.xyz.read_bytes())

    def test_stage2_partial_failure(self):
        with patch.object(cm, 'run_mlip_stage2', return_value=[
                ('ok', -100, self.pos, 2, False), ('error','test failure',None,0,False)]):
            result = self.pipeline(mlip={'stage1_only':False})
        self.assertEqual(result['status'], 'error')
        work = Path(result['work'])
        self.assertIn('STAGE2_MLIP_PARTIAL', (work / 'stage2_mlip.log').read_text())
        self.assertFalse(json.loads((work / 'optimization_status.json').read_text())[0]['converged'])

    def test_worker_sp_and_constrained_fire(self):
        from ase.calculators.calculator import Calculator, all_changes
        class Quadratic(Calculator):
            implemented_properties = ['energy', 'forces']
            def calculate(self, atoms=None, properties=None, system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                p = atoms.positions
                self.results = {'energy':float((p*p).sum()/2), 'forces':-p.copy()}
                self.info = dict(atoms.info)
        calc = Quadratic()
        rq = queue.Queue()
        with patch.dict(cm._worker_state, {'charge':0, 'spin':1, 'rq':rq}, clear=True), patch.object(cm, '_get_calc', return_value=calc):
            cm._compute_conformer((0, self.syms, self.pos, 'sp', 0.04, 2, [0]))
            _, sp = rq.get_nowait()
            cm._compute_conformer((1, self.syms, self.pos, 'opt', 0.04, 2, [0]))
            _, opt = rq.get_nowait()
        self.assertEqual(sp[0], 'ok')
        self.assertEqual(sp[2], self.pos)
        self.assertEqual(opt[3:], (2, False))
        self.assertEqual(opt[2][0], self.pos[0])
        self.assertLess(opt[1], sp[1])
        self.assertEqual(calc.info, {'charge':0, 'spin':1})

    def test_checkpoint_resume_and_retry(self):
        frames = [(self.syms, self.pos)] * 2
        cfg = dict(self.load()['mlip'], work=str(self.root / 'checkpoint'))
        rq = queue.Queue()
        class Pool:
            def __init__(self): self.calls = 0
            def apply_async(pool, fn, args):
                pool.calls += 1
                idx = args[0][0]
                payload = ('error','retry',None,0,False) if pool.calls == 2 else ('ok', -100, self.pos, 0, True)
                rq.put((idx, payload))
        pool = Pool()
        args = (frames, cfg, [pool], rq, threading.Lock(), queue.Queue(), 1, 'test')
        results = cm.run_mlip_stage2(*args)
        self.assertEqual(results[1][0], 'error')
        self.assertTrue(all(r[0] == 'ok' for r in cm.run_mlip_stage2(*args)))
        self.assertEqual(pool.calls, 3)
        cm.run_mlip_stage2(*args)
        self.assertEqual(pool.calls, 3)
        with self.assertRaisesRegex(ValueError, 'does not match'):
            cm.run_mlip_stage2(frames[:1], *args[1:])

    def test_invalid_checkpoint_payload(self):
        for payload in [('ok',float('nan'),self.pos,0,True), ('ok',1,[],0,True),
                        ('ok',1,self.pos,0,'true'), ('error','x',[],0,False)]:
            with self.subTest(payload=payload[:2]), self.assertRaises(ValueError):
                cm.validate_stage2_result(payload, self.syms)

    def test_worker_crash_detected(self):
        from types import SimpleNamespace
        with self.assertRaisesRegex(RuntimeError, 'worker exited'):
            cm.wait_gpu_result(queue.Queue(), [SimpleNamespace(exitcode=-9)])

    def test_cli_preflight(self):
        # Keep the output outside the input directory, as required by the CLI.
        path = self.root / 'config.yaml'
        self.cfg['output'] = str(self.root.parent / (self.root.name + '_output'))
        path.write_text(yaml.safe_dump(dict(self.cfg, dry_run=True)))
        with patch('sys.argv', ['crest_mace.py', str(path)]), patch.object(cm.shutil, 'which', return_value='/fake/crest'), patch.object(cm, 'detect_crest_version', return_value=(3, 0)), contextlib.redirect_stdout(io.StringIO()) as captured:
            self.assertEqual(cm.main(), 0)
        self.assertIn('DRY-RUN', captured.getvalue())
        self.assertFalse(Path(self.cfg['output']).exists())
        self.cfg['freeze'] = [133]
        path.write_text(yaml.safe_dump(dict(self.cfg, dry_run=True)))
        with patch('sys.argv', ['crest_mace.py', str(path)]), patch.object(cm.shutil, 'which', return_value='/fake/crest'), patch.object(cm, 'detect_crest_version', return_value=(3, 0)), contextlib.redirect_stdout(io.StringIO()) as captured:
            self.assertEqual(cm.main(), 1)
        self.assertIn('原子索引超出', captured.getvalue())


if __name__ == '__main__':
    unittest.main()
