'''
Created on Apr 11, 2018

@author: ywz
'''
import numpy, random
from collections import deque


class ReplayMemory:
    
    def __init__(self, history_len=4, capacity=1000000):
        
        self.capacity = capacity
        self.history_length = history_len
        
        self.states = deque([])
        self.others = deque([])
    
    def add(self, state, action, r, termination):
        
        if len(self.states) == self.capacity:
            self.states.popleft()
            self.others.popleft()
        self.states.append(state)
        self.others.append((action, r, termination))
        
    def add_nullops(self, init_state):
        for _ in range(self.history_length):
            self.add(init_state, 0, 0, 0)
    
    def phi(self, new_state):
        assert len(self.states) > self.history_length
        states = [new_state] + [self.states[-1-i] for i in range(self.history_length-1)]
        return numpy.concatenate(states, axis=0)
    
    def _phi(self, index):
        states = [self.states[index-i] for i in range(self.history_length)]
        return numpy.concatenate(states, axis=0)
    
    def sample(self):
        
        while True:
            
            index = random.randint(a=self.history_length-1, b=len(self.states)-2)
            infos = [self.others[index-i] for i in range(self.history_length)]
            # Check if termination=1 before "index"
            flag = False
            for i in range(1, self.history_length):
                if infos[i][2] == 1:
                    flag = True
                    break
            if flag:
                continue
            
            state = self._phi(index)
            new_state = self._phi(index+1)
            action, r, termination = self.others[index]
            state = numpy.asarray(state, dtype=numpy.float32)
            new_state = numpy.asarray(new_state, dtype=numpy.float32)
                
            return (state, action, r, new_state, termination)


if __name__ == "__main__":
    
    history_len = 2
    capacity = 20
    
    replay = ReplayMemory(history_len, capacity)
    
    for i in range(20):
        state = numpy.zeros((2,)) + i
        action = numpy.ones((2,)) * i
        reward = i ** 2
        termination = 1 if i % 10 == 0 else 0
        replay.add(state, action, reward, termination)
        
    print(replay.states)
    print(replay.others)
    state, action, r, new_state, termination = replay.sample()
    print(state)
    print(new_state)
    print(action)
    print(r)
    print(termination)
    print('------------------------------')
    
    for _ in range(50):
        replay.sample()
        
