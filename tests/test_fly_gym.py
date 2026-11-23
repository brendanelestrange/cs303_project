"""Fast tests for fly_games.py and the learners in fly_gym.py (no Julia needed).

Run: python -m unittest discover tests
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fly_games as fg
import fly_gym as fy


class Observations(unittest.TestCase):
    def test_every_state_of_a_discrete_space_looks_different(self):
        mask, enc = fg.encoder(spaces.Discrete(7, start=3))
        codes = {tuple(enc(s)) for s in range(3, 10)}
        self.assertEqual(len(codes), 7)
        self.assertEqual(len(mask), 4)
        self.assertTrue(all(np.abs(c).sum() == 1 for c in codes))

    def test_nested_spaces_are_concatenated(self):
        space = spaces.Dict(a=spaces.Box(-1, 1, (2, 3)), b=spaces.Tuple([spaces.Discrete(5), spaces.MultiBinary(2)]),
                            c=spaces.MultiDiscrete([3, 2]))
        mask, enc = fg.encoder(space)
        for _ in range(20):
            self.assertEqual(enc(space.sample()).shape, mask.shape)
        self.assertEqual(mask.sum(), 6)

    def test_images_are_refused(self):
        with self.assertRaises(ValueError):
            fg.encoder(spaces.Box(0, 255, (96, 96, 3), np.uint8))


class Actions(unittest.TestCase):
    def test_every_choice_is_a_legal_action(self):
        rng = np.random.default_rng(0)
        for space in (spaces.Discrete(4, start=-1), spaces.MultiDiscrete([3, 2]), spaces.MultiBinary(3),
                      spaces.Box(-2, 2, (1,)), spaces.Box(np.array([-1, 0]), np.array([1, 5])),
                      spaces.Box(-np.inf, np.inf, (2,))):
            heads, act = fg.action_heads(space, 5)
            for _ in range(50):
                a = act([int(rng.integers(n)) for n in heads])
                self.assertTrue(space.contains(a) or space.contains(np.int64(a)), (space, a))

    def test_continuous_choices_reach_both_ends(self):
        _, act = fg.action_heads(spaces.Box(np.array([-1, 0]), np.array([1, 5])), 3)
        self.assertEqual(act([0, 2]).tolist(), [-1, 5])


class Games(unittest.TestCase):
    def test_presets_stay_in_range(self):
        rng = np.random.default_rng(0)
        for name in fg.PRESETS:
            with self.subTest(name):
                try:
                    g = fg.Game(name)
                except Exception as e:                       # box2d or flappy-bird-gymnasium not installed
                    self.skipTest(f"{name}: {e}")
                f, done = g.reset(0), False
                for _ in range(300):
                    self.assertEqual(f.shape, (g.n_obs,))
                    self.assertTrue(np.all(np.abs(f) <= 1))
                    f, r, bonus, term, trunc = g.step(g.random_choice(rng))
                    if term or trunc:
                        break

    def test_games_without_a_time_limit_still_end(self):
        g = fg.Game("cliffwalking")
        g.reset(0)
        for t in range(200):
            *_, term, trunc = g.step([0])                    # walk up forever
        self.assertTrue(trunc and not term)

    def test_unknown_game(self):
        with self.assertRaises(ValueError):
            fg.Game("nosuchgame")


class Learning(unittest.TestCase):
    def test_advantages(self):
        # ends for real at t=1 (no future), then a time-out at t=2 (future kept)
        adv = fy.advantages([1, 1, 1], [0, 0, 0], [5, 5, 5], [False, True, False], [False, True, True], 0.5, 1.0)
        np.testing.assert_allclose(adv, [1 + 2.5 + 0.5 * 1, 1, 1 + 2.5])

    def test_ppo_gradient_direction(self):
        rng = np.random.default_rng(0)
        ro = fy.Readout(2, [3, 2], np.zeros(2), np.ones(2))
        X = np.column_stack([rng.normal(size=(400, 2)), np.ones(400)])
        C = np.column_stack([rng.integers(3, size=400), rng.integers(2, size=400)])
        adv = (C[:, 0] == 2) + (C[:, 1] == 0) * 1.0          # reward choice 2 on head 1 and choice 0 on head 2
        args = fy.parse_args(["train", "cartpole"])
        fy.ppo_update(ro, fy.Adam(ro.w.shape), fy.Adam(ro.v.shape), X, C, adv, adv, args, 0.01, rng)
        self.assertEqual(ro.choose(np.array([0.0, 0, 1]), rng, greedy=True), [2, 0])

    def test_no_brain_learns_saves_and_reloads(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "p.npz")
            args = fy.parse_args(["train", "CartPole-v1", "--no-brain", "--steps", "12000", "--lr", "0.02", "--save", path])
            game = fy.make_game(args.game)
            game.calibrate()
            player = fy.NoBrain(game.n_obs, 2)
            policy, log = fy.train_readout(args, game, player, lambda p, log: None)
            policy["ro"].greedy = False
            fy.write_player(path, args, game, player, policy, log)
            before, _ = fy.play_frozen(game, player, policy, 10)
            after, _ = fy.play_frozen(*fy.read_player(path, False), 10)
            np.testing.assert_array_equal(before, after)
            self.assertGreater(before.mean(), 60)            # random play gets about 22


if __name__ == "__main__":
    unittest.main()
