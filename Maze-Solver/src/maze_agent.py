'''
BlindBot MazeAgent meant to employ Propositional Logic,
Search, Planning, and Active Learning to navigate the
Maze Pitfall problem
'''

from asyncio import queues
from importlib.machinery import PathFinder
import time
import random
from maze_clause import MazeClause
from pathfinder import *
from maze_problem import *
from queue import Queue
from maze_knowledge_base import MazeKnowledgeBase
from constants import Constants

# [!] TODO: import your Problem 1 when ready here!

class MazeAgent:
    
    ##################################################################
    # Constructor
    ##################################################################
    
    def __init__ (self, env, perception):
        """
        Initializes the MazeAgent with any attributes it will need to
        navigate the maze
        :env: The Environment in which the agent is operating
        :perception: The starting perception of the agent, which is a
        small dictionary with keys:
          - loc:  the location of the agent as a (c,r) tuple
          - tile: the type of tile the agent is currently standing upon
        """
        self.env  = env
        self.loc  = env.get_player_loc()
        self.goal = env.get_goal_loc()
        
        # The agent's maze can be manipulated as a tracking mechanic
        # for what it has learned; changes to this maze will be drawn
        # by the environment and is used for visuals and pathfinding
        self.maze = env.get_agent_maze()
        
        
        # The agent's plan will be a queue storing the sequence of
        # actions that the environment will execute
        self.plan = Queue()
        
        self.kb = MazeKnowledgeBase()
        self.kb.tell(MazeClause([(("P", self.goal), False)]))
        self.kb.tell(MazeClause([(("P", self.loc), False)]))
        self.unexplored_tiles = env.get_playable_locs()
        self.unexplored_tiles.remove(self.goal)
        self.visited_tiles = [] 
        self.safe_tiles = set()

        
    
    
    ##################################################################
    # Methods
    ##################################################################
    
    def is_safe_tile (self, loc):
        """
        Can be used in the think method or unit tests for determining whether
        or not a tile given in the provided location is safe (i.e., sans Pit).
        :loc: A tuple (c,r) with a presumably valid maze location that is within
              the non-wall bounds of the current maze
        :returns: One of three return values:
              1. True if the location is certainly safe (i.e., not pit)
              2. False if the location is certainly dangerous (i.e., pit)
              3. None if the safety of the location cannot be currently determined
        """
        if self.kb.ask(MazeClause([(("P", loc), True)])): return False
        elif self.kb.ask(MazeClause([(("P", loc), False)])): return True
        else: return None

    
    def think(self, perception):
        """
        think is parameterized by the agent's perception of the tile type
        on which it is now standing, and is called during the environment's
        action loop. This method is the chief workhorse of your MazeAgent
        such that it must then generate a plan of action from its current
        knowledge about the environment.
        
        :perception: A dictionary providing the agent's current location
        and current tile type being stood upon, of the format:
          {"loc": (x, y), "tile": tile_type}
        """
        mp = MazeProblem(self.maze)   
        if perception["loc"] in self.unexplored_tiles : 
            self.plan = Queue()
            self.unexplored_tiles.remove(perception["loc"])
        MazeAgent.add_to_KB(self, perception)
        if perception["loc"] == self.goal : return
        self.visited_tiles.append(perception["loc"])
        card_locs = self.env.get_cardinal_locs(perception["loc"], 1)
        playable_locs = []
        for loc in card_locs: 
            if loc in self.env.get_playable_locs(): playable_locs.append(loc) 
            if loc == self.goal :
                direction = list(Constants.MOVE_DIRS.keys())[list(Constants.MOVE_DIRS.values()).index((loc[0] - perception["loc"][0], loc[1] - perception["loc"][1]))]
                self.plan = Queue()
                self.plan.put(direction)
                return
        safe_unexplored = self.unexplored_tiles.intersection(self.safe_tiles)
        best_loc = list(safe_unexplored).pop() if len(safe_unexplored) > 0 else playable_locs[0]
        for loc in safe_unexplored :
            if heuristic(loc, self.goal) < heuristic(best_loc, self.goal) : 
                best_loc = loc
        creative_track_tiles = pathfind(mp, perception["loc"], best_loc)
        for direction in creative_track_tiles[1] :
            self.plan.put(direction)
        return
 

        
        
    def add_to_KB(self,perception): 
        """
        Adds to the knowledge base depending on the tile the agent is currently on. 
        agent can infer that differing spaces are safe due to the tile its on (example: 
        "3" state guaratnees the cardinal spaces 2 away are safe)
        :perception: A dictionary providing the agent's current location
        and current tile type being stood upon, of the format:
          {"loc": (x, y), "tile": tile_type}
        """
        card_locs = []
        state = perception["tile"]
        if state == "." :
            self.maze[perception["loc"][1]][perception["loc"][0]] = "S"
            self.kb.tell(MazeClause([(("P", perception["loc"]), False)]))
            for loc in self.env.get_cardinal_locs(perception["loc"], 4) :
                if loc in self.env.get_playable_locs():
                    if self.is_safe_tile(loc) == False : self.maze[loc[1]][loc[0]] = "P"
            for loc in self.env.get_cardinal_locs(perception["loc"], 3) :
                if loc in self.env.get_playable_locs():
                    self.maze[loc[1]][loc[0]] = "S"
                    self.safe_tiles.add(loc)
                    card_locs.append(loc)
            for loc in self.env.get_cardinal_locs(perception["loc"], 2) :
                if loc in self.env.get_playable_locs():
                    self.maze[loc[1]][loc[0]] = "S"
                    self.safe_tiles.add(loc)
                    card_locs.append(loc)
            for loc in self.env.get_cardinal_locs(perception["loc"], 1) :
                if loc in self.env.get_playable_locs():
                    self.maze[loc[1]][loc[0]] = "S"
                    self.safe_tiles.add(loc)
                    card_locs.append(loc)
        if state == "3" :
            self.maze[perception["loc"][1]][perception["loc"][0]] = "S"
            self.kb.tell(MazeClause([(("P", perception["loc"]), False)]))
            for loc in self.env.get_cardinal_locs(perception["loc"], 3) :
                if loc in self.env.get_playable_locs():
                    if self.is_safe_tile(loc) == False : self.maze[loc[1]][loc[0]] = "P"
            for loc in self.env.get_cardinal_locs(perception["loc"], 2) :
                if loc in self.env.get_playable_locs():
                    self.maze[loc[1]][loc[0]] = "S" 
                    self.safe_tiles.add(loc)
                    card_locs.append(loc)
            for loc in self.env.get_cardinal_locs(perception["loc"], 1) :
                if loc in self.env.get_playable_locs():
                    self.maze[loc[1]][loc[0]] = "S" 
                    self.safe_tiles.add(loc)
                    card_locs.append(loc)
        if state == "2" :
            self.maze[perception["loc"][1]][perception["loc"][0]] = "S"
            self.kb.tell(MazeClause([(("P", perception["loc"]), False)]))
            for loc in self.env.get_cardinal_locs(perception["loc"], 2) :
                if loc in self.env.get_playable_locs():
                    if self.is_safe_tile(loc) == False : self.maze[loc[1]][loc[0]] = "P"
            for loc in self.env.get_cardinal_locs(perception["loc"], 1) : 
                if loc in self.env.get_playable_locs():
                    self.maze[loc[1]][loc[0]] = "S"
                    self.safe_tiles.add(loc)
                    card_locs.append(loc)
        if state == "P" :
            self.kb.tell(MazeClause([(("P", perception["loc"]), True)]))
        if state == "1" :
            self.maze[perception["loc"][1]][perception["loc"][0]] = "S"
            self.kb.tell(MazeClause([(("P", perception["loc"]), False)]))
            possible_pits = []
            for loc in self.env.get_cardinal_locs(perception["loc"], 1) :
                possible_pits.append((("P", loc), True))
            self.kb.tell(MazeClause([possible_pits]))
        else :
            for loc in card_locs :
                 self.kb.tell(MazeClause([(("P", loc), False)]))
    

    def get_next_move(self):
        """
        Returns the next move in the plan, if there is one, otherwise None
        [!] You should NOT need to modify this method -- contact Dr. Forney
            if you're thinking about it
        """
        return None if self.plan.empty() else self.plan.get()
    