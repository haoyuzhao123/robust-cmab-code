__author__ = 'http://docs.python.org/2/library/heapq.html#priority-queue-implementation-notes'

import itertools
from heapq import *

class PriorityQueue(object):
    def __init__(self):
        self.pq = []                         # list of entries arranged in a heap
        self.entry_finder = {}               # mapping of tasks to entries
        self.REMOVED = '<removed-task>'      # placeholder for a removed task
        self.counter = itertools.count()     # unique sequence count

    def add_task(self, task, priority=0, activated_nodes=[]):
        'Add a new task or update the priority of an existing task'
        if task in self.entry_finder:
            self.remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task, activated_nodes]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)

    def remove_task(self, task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def pop_item(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        self.pq.sort()
        while self.pq:
            priority, count, task, activated_nodes = heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task, priority, activated_nodes
        raise KeyError('pop from an empty priority queue')

    def __str__(self):
        return str([entry for entry in self.pq if entry[2] != self.REMOVED])


# pq = PriorityQueue()
# pq.add_task(pq.REMOVED, -100)
# pq.add_task(1, -75)
# pq.add_task(2, -50)
# pq.add_task(pq.REMOVED, -25)
if __name__ == '_main__':
    console = []