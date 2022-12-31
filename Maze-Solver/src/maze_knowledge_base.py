'''
maze_knowledge_base.py

Specifies a simple, Conjunctive Normal Form Propositional
Logic Knowledge Base for use in Grid Maze pathfinding problems
with side-information.
'''
from maze_clause import MazeClause
import unittest

class MazeKnowledgeBase:
    
    def __init__ (self):
        self.clauses = set()
    
    def tell (self, clause):
        """
        Adds the given clause to the CNF MazeKnowledgeBase
        Note: we expect that no clause added this way will ever
        make the KB inconsistent (you need not check for this)
        """
        self.clauses.add(clause)
        return
        
    def ask (self, query):
        """
        Given a MazeClause query, returns True if the KB entails
        the query, False otherwise
        """
        return self.PL_resolution( self.clauses, query)

    def PL_resolution(self, KB, alpha) :
        clauses = self.clauses.copy()
        for prop in alpha.props.items() : 
            clauses.add(MazeClause([(prop[0], not prop[1])]))
        new = set()
        while True :
            for clause_i in clauses :
                for clause_j in clauses :
                    if clause_i != clause_j :
                        resolvent = MazeClause.resolve(clause_i, clause_j)
                        if MazeClause([]) in resolvent: return True
                        new.update(resolvent)
            if new.issubset(clauses) : return False
            clauses = clauses.union(new)




class MazeKnowledgeBaseTests(unittest.TestCase):
    def test_mazekb1(self):
        kb = MazeKnowledgeBase()
        kb.tell(MazeClause([(("X", (1, 1)), True)]))
        self.assertTrue(kb.ask(MazeClause([(("X", (1, 1)), True)])))
        
    def test_mazekb2(self):
        kb = MazeKnowledgeBase()
        kb.tell(MazeClause([(("X", (1, 1)), False)]))
        kb.tell(MazeClause([(("X", (1, 1)), True), (("Y", (1, 1)), True)]))
        self.assertTrue(kb.ask(MazeClause([(("Y", (1, 1)), True)])))
        
    def test_mazekb3(self):
        kb = MazeKnowledgeBase()
        kb.tell(MazeClause([(("X", (1, 1)), False), (("Y", (1, 1)), True)]))
        kb.tell(MazeClause([(("Y", (1, 1)), False), (("Z", (1, 1)), True)]))
        kb.tell(MazeClause([(("W", (1, 1)), True), (("Z", (1, 1)), False)]))
        kb.tell(MazeClause([(("X", (1, 1)), True)]))
        self.assertTrue(kb.ask(MazeClause([(("W", (1, 1)), True)])))
        self.assertFalse(kb.ask(MazeClause([(("Y", (1, 1)), False)])))


if __name__ == "__main__":
    unittest.main()