from environment import *
from maze_clause import *
from maze_knowledge_base import *
from copy import deepcopy
import timeout_decorator

class MazeKBTests(unittest.TestCase):
    
# -----------------------------------------------------------------------------------------
# MazeInference Tests
# -----------------------------------------------------------------------------------------

    def test_inference0(self):
        #    c-> 012345   # r
        maze = ["XXXXXX", # 0
                "X...GX", # 1
                "X..PPX", # 2
                "X....X", # 3
                "X..P.X", # 4
                "X@...X", # 5
                "XXXXXX"] # 6
        env = Environment(maze, tick_length = 0, verbose = False)
        env.test_move(None)
        # Tiles 3-around a safe space without a warning signal should be known safe
        self.assertTrue(env.test_safety_check((1,5)))
        self.assertTrue(env.test_safety_check((1,4)))
        self.assertTrue(env.test_safety_check((1,3)))
        self.assertTrue(env.test_safety_check((1,2)))
        self.assertTrue(env.test_safety_check((2,5)))
        self.assertTrue(env.test_safety_check((3,5)))
        self.assertTrue(env.test_safety_check((4,5)))
        
        # If that's the only perception, however, other tiles won't be known safe
        self.assertEqual(None, env.test_safety_check((2,4)))
        self.assertEqual(None, env.test_safety_check((4,2)))
        
        # Still, the goal should always be safe
        self.assertTrue(env.test_safety_check((4,1)))
        
    def test_inference1(self):
        #    c-> 012345   # r
        maze = ["XXXXXX", # 0
                "X...GX", # 1
                "X..PPX", # 2
                "X....X", # 3
                "X..P.X", # 4
                "X..@.X", # 5
                "XXXXXX"] # 6
        env = Environment(maze, tick_length = 0, verbose = False)
        env.test_move(None)
        
        # Uh oh, a distance of 1 all around the start? Bad placement, won't know
        # where that pit is
        self.assertEqual(None, env.test_safety_check((3,4)))
        self.assertEqual(None, env.test_safety_check((4,5)))
        self.assertEqual(None, env.test_safety_check((2,5)))
        
        # But, suppose we get lucky and move left to safety...
        env.test_move("L")
        
        # Well, now we know where that Pit was
        self.assertFalse(env.test_safety_check((3,4)))
        self.assertTrue(env.test_safety_check((4,5)))
        self.assertTrue(env.test_safety_check((2,5)))
        

# -----------------------------------------------------------------------------------------
# MazeAgent Tests
# -----------------------------------------------------------------------------------------

MIN_SCORE = -100
NUM_ITERS = 5

def run_one_maze (maze):
    scores = [max(Environment(maze, tick_length = 0, verbose = False).start_mission(), -100) for iter in range(NUM_ITERS)]
    return sum(scores) / NUM_ITERS

def run_mazes (mazes, grading_threshold=-100, title="Maze Pitfall Tests"):
    """
    Runs and scores a list of mazes based on an average agent performance on each
    :param mazes: list of mazes to run the agent upon
    :param grading_threshold: threshold to do better-than in order to have "passed"
           the given maze
    :param title: string indicating the title of the given tests to print out 
    """
    # Twist: duplicate the input mazes... in reverse!
    print("----------------------------")
    print("[!] Tests Running: " + title)
    new_mazes = deepcopy(mazes)
    for m in new_mazes:
        m.reverse()
    mazes.extend(new_mazes)
    total, attempted, passed = 0, 0, 0
    for maze in mazes:
        attempted += 1
        maze_avg = 0
        try:
            maze_avg = run_one_maze(maze)
        except:
            print(" [X] Error on maze")
            maze_avg = -100
        total += maze_avg
        if maze_avg > grading_threshold:
            passed += 1
        else:
            print("  [X] Failed maze with score of " + str(maze_avg))
            print("\n".join(maze))
    print("  Total Score: " + str(total))
    print("----------------------------")
    return (total, attempted, passed)

def report_results (results_list, title):
    total = sum([x[0] for x in results_list])
    attempted = sum([x[1] for x in results_list])
    passed = sum([x[2] for x in results_list])
    print("=============================")
    print("[!] Final Report on " + title)
    print("  [>] Passed: " + str(passed) + " / " + str(attempted))
    print("  [>] Total score: " + str(total))
    print("=============================")

    
def full_game_mazes():
    mazes = [
        # Easy difficulty: Score > -20
        ["XXXXXX",
         "X...GX",
         "X..PPX",
         "X....X",
         "X..P.X",
         "X@...X",
         "XXXXXX"],
             
        # Medium difficulty: Score > -30
        ["XXXXXXXXX",
         "X..PGP..X",
         "X.......X",
         "X..P.P..X",
         "X.......X",
         "X..@....X",
         "XXXXXXXXX"],
             
        # Hard difficulty: Score > -40
        ["XXXXXXXXX",
         "X...G.PPX",
         "X...P...X",
         "X.......X",
         "XP......X",
         "XP.P.P.PX",
         "X...@...X",
         "XXXXXXXXX"]
    ]
    
    all_runs = []
    all_runs.append(run_mazes(mazes[0:1], -20, "Easy Grading Mazes"))
    all_runs.append(run_mazes(mazes[1:2], -30, "Medium Grading Mazes"))
    all_runs.append(run_mazes(mazes[2: ], -40, "Hard Grading Mazes"))
    return all_runs
        
if __name__ == "__main__":
    report_results(full_game_mazes(), "Test Mazes")
    timeout_decorator.timeout(2, use_signals=False)(unittest.main)()