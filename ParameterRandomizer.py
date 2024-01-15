import numpy as np
import random
import math
class ParameterRandomizer():
    def __init__(self):
        pass
    
    def gen_params_random(self, surface_facet=None, particle_surface_facet=None,particle_rotation_arg=True):
        # hull layers
        # interface radiis
        # layer sample points
        # centers
        # add_step
        # step_height
        # surface_facet
        # particle_surface_facet
        # particle_rotation
        hull_layers = random.randint(2,7)
        interface_radiis = []
        layer_sample_points = []
        centers = []
        wetting = random.choice(["acute","acute","obtuse","obtuse","90"])

        if wetting == "90":
            radius = random.randint(4,12)
            for i in range(hull_layers):
                interface_radiis.append(radius)
        
        if wetting == "acute":
            if hull_layers <= 0:
                raise ValueError("Invalid hull_layers value")
            #print(hull_layers)
            # Generate l-1 random numbers
            random_numbers = [random.randint(4,12) for _ in range(hull_layers)]
            # sort descending
            random_numbers.sort(reverse=True)
            #print(random_numbers)
            interface_radiis = random_numbers
                
        if wetting == "obtuse":
            if hull_layers <= 0:
                raise ValueError("Invalid hull_layers value")
            #print(hull_layers)
            # Generate l-1 random numbers
            random_numbers = [random.randint(4,12) for _ in range(hull_layers)]
            #print("random numbers",random_numbers)
            # sort descending
            random_numbers.sort(reverse=True)
            #print("sorted",random_numbers)
            if hull_layers <= 2:
                random_numbers.sort()
                interface_radiis = random_numbers
            else:
                if hull_layers ==3:
                    max_position = 2
                elif hull_layers ==4:
                    max_position = random.choice([2,3])
                else:
                    max_position = random.randint(2,math.floor(hull_layers/2))
                #print("max_position",max_position)
                reversed_sublist = random_numbers[:max_position][::-1]
                #print("reversed_sublist",reversed_sublist)
                random_numbers[:max_position] = reversed_sublist
                #print("random_numbers",random_numbers)
                interface_radiis = random_numbers
                
        
        for i in range(hull_layers):
            layer_sample_points.append(random.randint(4,10))
            centers.append([np.random.normal(0,0.5),np.random.normal(0,0.5)])

        
        add_step = random.choice([True,False])
        step_height = random.randint(1,hull_layers)
        if surface_facet == None:
            surface_facet = random.choice(["100","100","111","111","random"])
        if particle_surface_facet == None:
            particle_surface_facet = random.choice(["100","100","111","111","random"])
        rotation = random.choice([True,False])
        
        if particle_surface_facet == "100":
            particle_rotation = random.choice([0,np.pi/2])
        elif particle_surface_facet == "111":
            particle_rotation = random.choice([0,np.pi/3])
        else:
            particle_rotation = 0

        if rotation and particle_rotation_arg:
            particle_rotation = random.uniform(0,2*np.pi)

        #print("hull_layers",hull_layers)
        #print("interface_radiis",len(interface_radiis))
        #print("layer_sample_points",len(layer_sample_points))
        #print("centers",len(centers))

        dict_of_parameters = {"hull_layers":hull_layers,
                              "interface_radiis":interface_radiis,
                              "layer_sample_points":layer_sample_points,
                              "centers":centers,
                              "add_step":add_step,
                              "step_height":step_height,
                              "surface_facet":surface_facet,
                              "particle_surface_facet":particle_surface_facet,
                              "particle_rotation":particle_rotation}

        return dict_of_parameters
    
    def gen_params_wulff(self, surface_facet=None, particle_surface_facet=None,particle_rotation_arg=True):
        # size
        # add_step
        # step_height
        # surface_facet
        # particle_surface_facet
        # particle_rotation

        size = random.randint(25,325)
        add_step = random.choice([True,False])
        step_height = 2
        if surface_facet == None:
            surface_facet = random.choice(["100","100","111","111","random"])
        if particle_surface_facet == None:
            particle_surface_facet = random.choice(["100","111"])
        rotation = random.choice([True,False])

        if particle_surface_facet == "100":
            particle_rotation = random.choice([0,np.pi/2])
        elif particle_surface_facet == "111":
            particle_rotation = random.choice([0,np.pi/3])

        if rotation and particle_rotation_arg:
            particle_rotation = random.uniform(0,2*np.pi)

        dict_of_parameters = {"wulff_size":size,
                              "add_step":add_step,
                              "step_height":step_height,
                              "surface_facet":surface_facet,
                              "particle_surface_facet":particle_surface_facet,
                              "particle_rotation":particle_rotation}
        #print(dict_of_parameters)
        return dict_of_parameters
    
    def gen_params_cluster(self, surface_facet=None, particle_surface_facet=None,particle_rotation_arg=True):

        cluster_surfaces = [(1, 0, 0), (1, 1, 1), (1, -1, 1)]
        cluster_layers = [random.randint(2,6),random.randint(2,6),random.randint(-1,0)]
        if surface_facet == None:
            surface_facet = random.choice(["100","100","111","111","random"])
        rotation = random.choice([True,False])

        particle_rotation = random.choice([0,np.pi/3])

        if rotation and particle_rotation_arg:
            particle_rotation = random.uniform(0,2*np.pi)
            

        add_step = random.choice([True,False])
        step_height = 2

        dict_of_parameters = {"cluster_surfaces":cluster_surfaces,
                                "cluster_layers":cluster_layers,
                                "add_step":add_step,
                                "step_height":step_height,
                                "surface_facet":surface_facet,
                                "particle_surface_facet":"111",
                                "particle_rotation":particle_rotation}
        return dict_of_parameters

    def gen_params_AC(self,structure_type,j):
        decider = j%4
        if decider == 0:
            surface_facet = "111"
            particle_surface_facet = "111"
            if structure_type == "random":
                dict_of_parameters = self.gen_params_random(surface_facet,particle_surface_facet,particle_rotation_arg=False)
            elif structure_type == "wulff":
                dict_of_parameters = self.gen_params_wulff(surface_facet,particle_surface_facet,particle_rotation_arg=False)
            elif structure_type == "cluster":
                dict_of_parameters = self.gen_params_cluster(surface_facet,particle_surface_facet,particle_rotation_arg=False)
        elif decider == 1:
            surface_facet = "100"
            particle_surface_facet = "111"
            if structure_type == "random":
                dict_of_parameters = self.gen_params_random(surface_facet,particle_surface_facet,particle_rotation_arg=False)
            elif structure_type == "wulff":
                dict_of_parameters = self.gen_params_wulff(surface_facet,particle_surface_facet,particle_rotation_arg=False)
            elif structure_type == "cluster":
                dict_of_parameters = self.gen_params_cluster(surface_facet,particle_surface_facet,particle_rotation_arg=False)
        elif decider == 2:
            surface_facet = "111"
            particle_surface_facet = "100"
            if structure_type == "cluster":
                structure_type = random.choice(["random","wulff"])
            if structure_type == "random":
                dict_of_parameters = self.gen_params_random(surface_facet,particle_surface_facet,particle_rotation_arg=False)
            elif structure_type == "wulff":
                dict_of_parameters = self.gen_params_wulff(surface_facet,particle_surface_facet,particle_rotation_arg=False)

        elif decider == 3:
            surface_facet = "100"
            particle_surface_facet = "100"
            if structure_type == "cluster":
                structure_type = random.choice(["random","wulff"])
            if structure_type == "random":
                dict_of_parameters = self.gen_params_random(surface_facet,particle_surface_facet,particle_rotation_arg=False)
            elif structure_type == "wulff":
                dict_of_parameters = self.gen_params_wulff(surface_facet,particle_surface_facet,particle_rotation_arg=False)
        else:
            raise ValueError("Unknown structure type")
        return structure_type, dict_of_parameters



    def randomize_parameters(self,structure_type,j=0):
        if structure_type == "random":
            dict_of_parameters = self.gen_params_random()
        elif structure_type == "wulff":
            dict_of_parameters = self.gen_params_wulff()
        elif structure_type == "cluster":
            dict_of_parameters = self.gen_params_cluster()
        elif structure_type == "atom_counting":
            structure_type = random.choice(["random","wulff","cluster"])
            structure_type,dict_of_parameters = self.gen_params_AC(structure_type,j)
        else:
            raise ValueError("Unknown structure type")
        #print("Structure type:",structure_type)
        return structure_type,dict_of_parameters