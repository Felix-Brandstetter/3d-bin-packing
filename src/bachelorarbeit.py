import numpy as np
import binpack
import matplotlib.pyplot as plt
from binpack.item import Item
from binpack.bin3d import Bin
import networkx as nx
import time
import random


class Action():
    """This class represents an action that can be performed."""
    def __init__(self,position,rotation,item,probability):
        self.position = position
        self.rotation = rotation
        self.item = item
        self.probability = probability
        
    def __hash__(self):
        item_hash= self.item.physical_hash()
        
        return hash((
            tuple(self.position),
            self.rotation,

        ))


    
    
class Instance_of_3D_BPP():
    """This class represents an Instance of the three-dimensional bin packing problem."""
    def __init__(self,bin3d:Bin,targetitems:list):
        self.bin3d = bin3d
        self.targetitems = targetitems
        self.difficulty_exact = 0
        self.difficulty_estimated = 0
    
    def __hash__(self):
        return hash((
            tuple(self.targetitems),
            self.bin3d,
        ))
        
    def get_alpha_value(bin_liste:list):
        """Returns the alpha value for a given list of bins"""
        data=[]
        for bin3d in bin_liste:
            data.append(bin3d.volume_percentage())
        alpha_value = np.mean(data)   
        return alpha_value
    
    def get_possible_actions(item: Item, bin3d: Bin):
        """Returns all possible actions for a given item and a given bin"""
        possible_actions=[]
        rotations = list(range(6))
        for rotation in rotations:
                    item_temp =item.copy().rotate(rotation)
                    valid = bin3d.valid_map(item_temp)
                    if valid.any():
                        index=np.argwhere(valid)
                        for x,y in index:
                            z = bin3d.heightmap[x][y]
                            position= [x,y,z]
                            possible_actions.append(Action(position,rotation,item_temp,1))
        
        for action in possible_actions:
                        action.probability=(1/len(possible_actions))
      
        return possible_actions
        
        
    
    def _generate_all_terminal_states(Items,bin_list,current_bin=0,alpha_values=[],recursion_level=0,bins_used=[],actions_taken=[],probability_of_final_states=[]):
        """Generates all terminal states and returns the corresponding alpha values and probabilities"""
        
        new_bin_used=False
        possible_actions=Instance_of_3D_BPP.get_possible_actions(Items[recursion_level],bin_list[current_bin])
                        
        if not possible_actions:
            new_bin = Bin(bin_list[current_bin].extents)
            bin_list.append(new_bin)
            current_bin = current_bin +1
            new_bin_used=True
            possible_actions=Instance_of_3D_BPP.get_possible_actions(Items[recursion_level],bin_list[current_bin])
                   
        assert(len(possible_actions)>0)
        for action in possible_actions:                  
            assert bin_list[current_bin].add(action.item,action.position)
            actions_taken.append(action)
                 
            if(recursion_level+1 < len(Items)):
                Instance_of_3D_BPP._generate_all_terminal_states(Items,bin_list,current_bin,alpha_values,
                                      (recursion_level+1),bins_used,actions_taken,probability_of_final_states)
                bin_list[current_bin].undo_add()   
                actions_taken.pop()
            
            else:
                alpha_value = Instance_of_3D_BPP.get_alpha_value(bin_list)
                alpha_values.append(alpha_value)
                probability_of_final_state=1
                for action in actions_taken:
                    probability_of_final_state=(probability_of_final_state*action.probability)
                probability_of_final_states.append(probability_of_final_state)
                bins_used.append(len(bin_list))
                bin_list[current_bin].undo_add()
                actions_taken.pop()
        
        if new_bin_used:
            bin_list.pop()
            
        return alpha_values, bins_used,probability_of_final_states,
    
    def calculate_difficulty_exact(self):
        """Returns the exact difficulty of an instance"""
        alpha_values = []
        bin_copy = self.bin3d.copy()
        targetitems_copy = self.targetitems.copy()
        
        alpha_values,bins_used,probability_of_final_states = Instance_of_3D_BPP._generate_all_terminal_states(targetitems_copy,                                                                    [bin_copy],0,[],0,[],[],[])
        expected_alpha_value = np.sum(np.multiply(alpha_values,probability_of_final_states))
        self.difficulty_exact = expected_alpha_value/ np.max(alpha_values)

        print(str(len(alpha_values))+ " final states were created")
        print("In the best case " + str(np.min(bins_used)) + " Bins were used")
        print("In the worst case " +str(np.max(bins_used)) +" Bins were use")
        print("The exact difficulty is: " +str(self.difficulty_exact))
        print("Maximum alpha value: " +str(np.max(alpha_values)))
        print("Minimum alpha value: " +str(np.min(alpha_values)))
        return self.difficulty_estimated
    
    def _generate_sample(items,bin3d,q):
        """Generates a sample of terminal states and returns the corresponding alpha values and probabilities"""
        alpha_values=[]
        probability_of_final_states=[]
        checked_terminal_states=[]
        bins_used=[]
        while (len(checked_terminal_states)<q):
            bin_list=[bin3d.copy()]
            current_bin =0
            actions_taken=[]

            for item in items:
                possible_actions=[]
                rotations = list(range(6))
                possible_actions = Instance_of_3D_BPP.get_possible_actions(item,bin_list[current_bin]) 
               
                if not possible_actions:
                    new_bin = Bin(bin_list[current_bin].extents)
                    bin_list.append(new_bin)
                    current_bin = current_bin +1
                    possible_actions = Instance_of_3D_BPP.get_possible_actions(item,bin_list[current_bin])        
                       
                assert(len(possible_actions)>0)
                action = random.choice(possible_actions)
                assert bin_list[current_bin].add(action.item,action.position)
                actions_taken.append(action)
    
            alpha_values.append(Instance_of_3D_BPP.get_alpha_value(bin_list))
            checked_terminal_states.append(hash(tuple(actions_taken)))
            probability_of_final_state=1
            for action in actions_taken:
                probability_of_final_state=(probability_of_final_state*action.probability)          
            probability_of_final_states.append(probability_of_final_state)
            bins_used.append(len(bin_list)) 
    
        return alpha_values,probability_of_final_states,bins_used
    
    
    
    def estimate_difficulty(self,sample_size):
        """Returns the estimated difficulty of the instance"""
        alpha_values,probability_of_final_states, bins_used=Instance_of_3D_BPP._generate_sample(self.targetitems,self.bin3d ,sample_size)

        expected_alpha_value = np.mean(alpha_values)
        self.difficulty_estimated = (expected_alpha_value)/(np.max(alpha_values))


        print(str(len(alpha_values))+ " final states were created")
        print("In the best case " + str(np.min(bins_used)) + " Bins were used")
        print("In the worst case " +str(np.max(bins_used)) +" Bins were use")
        print("The estimated difficulty is: " +str(self.difficulty_estimated))
        print("Maximum alpha value: " +str(np.max(alpha_values)))
        print("Minimum alpha value: " +str(np.min(alpha_values))) 
    
        return self.difficulty_estimated, np.max(alpha_values)
    
    
    def analyse(self,sample_size):
        course_of_difficulty=[]
        bin3d = self.bin3d.copy()
        items = self.targetitems.copy()
        alpha_values=[]
        probability_of_final_states=[]
        checked_terminal_states=[]
        bins_used=[]
        while (len(checked_terminal_states)<sample_size):
            bin_list=[bin3d.copy()]
            current_bin =0
            actions_taken=[]

            for item in items:
                possible_actions=[]
                rotations = list(range(6))
                possible_actions = Instance_of_3D_BPP.get_possible_actions(item,bin_list[current_bin]) 
               
                if not possible_actions:
                    new_bin = Bin(bin_list[current_bin].extents)
                    bin_list.append(new_bin)
                    current_bin = current_bin +1
                    possible_actions = Instance_of_3D_BPP.get_possible_actions(item,bin_list[current_bin])        
                       
                assert(len(possible_actions)>0)
                action = random.choice(possible_actions)
                assert bin_list[current_bin].add(action.item,action.position)
                actions_taken.append(action)
    
            alpha_values.append(Instance_of_3D_BPP.get_alpha_value(bin_list))
            checked_terminal_states.append(hash(tuple(actions_taken)))
            probability_of_final_state=1
            for action in actions_taken:
                probability_of_final_state=(probability_of_final_state*action.probability)          
            probability_of_final_states.append(probability_of_final_state)
            bins_used.append(len(bin_list)) 
            expected_alpha_value = np.mean(alpha_values)
            self.difficulty_estimated = (expected_alpha_value)/(np.max(alpha_values))
            course_of_difficulty.append(self.difficulty_estimated)
        
        plt.plot(course_of_difficulty,label='Difficulty estimated')
        plt.xlabel("|Q|")
        plt.ylabel("Difficulty")
        plt.title("Course of estimated difficulty")
        plt.legend() 
        plt.savefig("Bilder/analyse.jpg")
        plt.show()
        
        
    def _create_graph(Items,bin_list,current_bin=0,recursion_level=0,actions_taken=[],current_node="Root", G=nx.DiGraph()):
        """Creates a graph of all terminal states to be reached within an instance"""

        new_bin_used=False
        possible_actions=Instance_of_3D_BPP.get_possible_actions(Items[recursion_level],bin_list[current_bin])

        if not possible_actions:
            new_bin = Bin(bin_list[current_bin].extents)
            bin_list.append(new_bin)
            current_bin = current_bin +1
            new_bin_used=True
            possible_actions=Instance_of_3D_BPP.get_possible_actions(Items[recursion_level],bin_list[current_bin])

        assert(len(possible_actions)>0)
        for action in possible_actions:                  
            assert bin_list[current_bin].add(action.item,action.position)
            actions_taken.append(action)
            next_node=str(hash(tuple(actions_taken)))
            G.add_edge(str(current_node),str(next_node), label="Item an Position " + str(action.position) + " in Rotation: "                                            +str(action.rotation) + " in Bin " +str(current_bin) +" platzieren")


            if(recursion_level+1 < len(Items)):
                Instance_of_3D_BPP._create_graph(Items,bin_list,current_bin,(recursion_level+1),actions_taken,next_node, G)
                bin_list[current_bin].undo_add()   
                actions_taken.pop()

            else:
                bin_list[current_bin].undo_add()
                actions_taken.pop()

        if new_bin_used:
            bin_list.pop()

        return G
    
    
    def save_graph(self):
        """Saves a png file of a graph which shows all terminal states to be reached within an instance"""
        bin_copy = self.bin3d.copy()
        targetitems_copy = self.targetitems.copy()
        G = Instance_of_3D_BPP._create_graph(targetitems_copy,[bin_copy],0,0,[],"Root", nx.DiGraph())
        A = nx.nx_agraph.to_agraph(G)
        A.draw('Bilder/States.png',args = '-Gnodesep=1 -Gfont_size=0.1 -Granksep=10', prog = 'dot' ) 
        print("Saved Graph in Bilder/")
        return

    
def generate_random_Instance_of_3D_BPP(bin_size = [8,8,8], number_of_targetitems=10):
    """Returns a randomly created instance"""
    rng = np.random.default_rng(np.random.randint(1000))
    bin3d = Bin([bin_size[0],bin_size[1],bin_size[2]])              
    targetitems = []
    while(len(targetitems)<number_of_targetitems):
        length = rng.integers(1,bin3d.extents[0]+1)
        width = rng.integers(1,bin3d.extents[1]+1)
        height = rng.integers(1,bin3d.extents[2]+1)
        extents_item = [length,width,height]
        item = Item(extents_item)
        targetitems.append(item)

    return Instance_of_3D_BPP(bin3d ,targetitems) 
    
    