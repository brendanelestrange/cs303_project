"""Fast tests for fly_viewer.py (no Julia needed): every game has names, frames and a working game loop.

Run: python -m unittest discover tests
"""
import base64
import json
import sys
import unittest
from pathlib import Path

import numpy as np
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fly_games as fg
import fly_viewer as fv


def open_game(test, name, **kw):
    try:
        return fg.Game(name, **kw)
    except Exception as e:                                   # box2d or flappy-bird-gymnasium not installed
        test.skipTest(f"{name}: {e}")


class FakeFly(fv.Fly):
    """Random choices, no brain."""
    steps, dt = 80, 0.5

    def __init__(self, game):
        self.game, self.rng, self.learned = game, np.random.default_rng(0), []

    def reset(self):
        pass

    def decide(self, f):
        choices = self.game.random_choice(self.rng)
        return choices, np.zeros(sum(self.game.heads)), np.array([3, 1]), np.array([1, 1])

    def learn(self, r, bonus, f, term, done):
        self.learned.append(done)


class Names(unittest.TestCase):
    def test_every_preset_names_its_inputs_and_choices(self):
        for name in fg.PRESETS:
            with self.subTest(name):
                g = open_game(self, name)
                self.assertEqual(len(g.senses), g.n_obs)
                self.assertTrue(all(len(pair) == 2 for pair in g.senses))
                self.assertEqual([len(head) for head in g.moves], g.heads)

    def test_grid_tile_names_match_the_input_code(self):
        for name, columns in (("frozenlake", 4), ("cliffwalking", 12)):
            g = open_game(self, name)
            for state in range(g.env.observation_space.n):
                f = g.encode(state)
                k = int(np.flatnonzero(f)[0])
                self.assertEqual(g.senses[k][int(f[k] > 0)], f"row {state // columns + 1} col {state % columns + 1}")

    def test_automatic_names_cover_every_space(self):
        space = spaces.Dict(a=spaces.Box(-1, 1, (2, 3)), b=spaces.Tuple([spaces.Discrete(5), spaces.MultiBinary(2)]),
                            c=spaces.MultiDiscrete([3, 2]))
        self.assertEqual(len(fg.sense_names(space)), len(fg.encoder(space)[0]))
        self.assertEqual(fg.sense_names(spaces.Discrete(3, start=5)), [("state 6", "state 5"), ("", "state 7")])
        for space in (spaces.Discrete(4, start=-1), spaces.MultiDiscrete([3, 2]), spaces.MultiBinary(3),
                      spaces.Box(-2, 2, (1,)), spaces.Box(np.array([-1, 0]), np.array([1, 5]))):
            self.assertEqual([len(h) for h in fg.move_names(space, 5)], fg.action_heads(space, 5)[0])
        self.assertEqual(fg.move_names(spaces.Box(-2, 2, (1,)), 3), [["-2", "+0", "+2"]])


class Groups(unittest.TestCase):
    def test_colours_never_run_out(self):
        self.assertEqual(fv.action_groups([["a", "b", "c"]]), ([0, 1, 2], ["a", "b", "c"]))
        self.assertEqual(fv.action_groups([["a"] * 5] * 2), ([0] * 5 + [1] * 5, ["action 0", "action 1"]))
        for moves in ([["x"] * 18], [["x"] * 5] * 9):
            group_of, groups = fv.action_groups(moves)
            self.assertEqual((set(group_of), len(groups)), ({0}, 1))
            self.assertEqual(len(group_of), sum(map(len, moves)))

    def test_readout_neurons_go_to_the_choice_they_push(self):
        # one head of 3 and one of 2; neuron 0 pushes choice 1 of head 0, neuron 1 pushes choice 0 of head 1
        w = np.array([[0.0, 5.0], [2.0, 5.0], [0.0, 5.0], [0.0, 1.0], [0.0, -1.0]])
        best, strength = fv.pushed_choice(w, [slice(0, 3), slice(3, 5)])
        self.assertEqual(best.tolist(), [1, 3])
        np.testing.assert_allclose(strength, [4 / 3, 1])


class Playing(unittest.TestCase):
    def test_every_preset_plays_with_pictures(self):
        for name in fg.PRESETS:
            with self.subTest(name):
                fly = FakeFly(open_game(self, name, render="rgb_array"))
                show = fv.Show(fly, fv.Hub(), 0)
                first = show.tick()
                self.assertTrue(base64.b64decode(first["start"]).startswith(b"\x89PNG"))
                self.assertEqual(len(first["frames"]), fly.game.preset.repeat if not first["done"] else len(first["frames"]))
                for _ in range(40):
                    msg = show.tick()
                    self.assertTrue(msg["frames"] and msg["seconds"] > 0)
                    self.assertEqual(msg["start"] is not None, msg["decision"] == 1)
                    json.dumps(msg)
                self.assertEqual(sum(fly.learned), len(show.scores))
                fly.game.close()

    def test_returns_add_up_over_a_game(self):
        show = fv.Show(FakeFly(fg.Game("cartpole")), fv.Hub(), 0)                # also: a game without pictures
        msgs = [show.tick()]
        while not msgs[-1]["done"]:
            msgs.append(show.tick())
        self.assertEqual(msgs[-1]["score"], sum(m["reward"] for m in msgs))
        self.assertEqual((msgs[-1]["best"], msgs[-1]["mean"], msgs[0]["best"]), (len(msgs), len(msgs), None))
        self.assertEqual(show.tick()["game"], 2)
        self.assertEqual((msgs[0]["frames"], msgs[0]["start"]), ([], None))


class Controls(unittest.TestCase):
    def test_only_listed_players_can_be_opened(self):
        hub = fv.Hub({"cartpole": "players/cartpole.npz"})
        with self.assertRaises(ValueError):
            hub.control("player", "../../secret")
        hub.control("player", "cartpole")
        self.assertEqual(hub.wait_turn(), "switch")

    def test_gap_only_exists_for_games_with_pipes(self):
        hub = fv.Hub()
        hub.control("gap", 120)
        self.assertIsNone(hub.state()["gap"])
        hub.gap = 150
        hub.control("gap", 9999)
        self.assertEqual(hub.state()["gap"], 220)

    def test_saved_players_are_found(self):
        self.assertEqual(set(fv.find_players(fv.parse_args([]))), set(fg.PRESETS))


if __name__ == "__main__":
    unittest.main()
