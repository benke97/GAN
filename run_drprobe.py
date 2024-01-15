#%%
import pyautogui
import time
import pickle as pkl
import numpy as np
import pandas as pd

class DrProbeRunner():
    def __init__(self):
        self.dataset=2000
        self.confidence_level = 0.9

    def get_file(self,i):
        if i < 10:
            return "data_000"+str(i)+".cif"
        elif i < 100:
            return "data_00"+str(i)+".cif"
        elif i < 1000:
            return "data_0"+str(i)+".cif"
        else:
            return "data_"+str(i)+".cif"

    def click(self,image,confidence,number_of_clicks=1):
        try:
            button_location = pyautogui.locateOnScreen(image, confidence=confidence)
            if button_location is not None:
                button_point = pyautogui.center(button_location)
                pyautogui.click(button_point,clicks=number_of_clicks)
            #else:
            #    print(image+" could not be found.")
        except pyautogui.ImageNotFoundException:
            #print(image+" could not be found.")
            pass

    def default_mouse(self):
        pyautogui.moveTo(400,400)

    def click_coordinate(self,x,y):
        pyautogui.moveTo(x,y)
        pyautogui.click()

    def reset_dropdown(self):
        pyautogui.press('up',presses=50)

    def fill_calc_setup(self,offset_x="1.5",offset_y="1.5",size_x="3.9",size_y="3.9"):
        #get coordinates of images "horizontal" and "vertical" and "offset" and "size" and "sampling"
        self.wait_for_button('ims/horizontal.png')
        horizontal_location = pyautogui.locateOnScreen("ims/horizontal.png", confidence=0.95)
        self.wait_for_button('ims/vertical.png')
        vertical_location = pyautogui.locateOnScreen("ims/vertical.png", confidence=0.95)
        self.wait_for_button('ims/offset.png')
        offset_location = pyautogui.locateOnScreen("ims/offset.png", confidence=0.95)
        self.wait_for_button('ims/size.png')
        size_location = pyautogui.locateOnScreen("ims/size.png", confidence=0.95)
        self.wait_for_button('ims/sampling.png')
        sampling_location = pyautogui.locateOnScreen("ims/sampling.png", confidence=0.95)

        self.click_coordinate(offset_location[0]+offset_location[2]/2,horizontal_location[1]+horizontal_location[3]/2)
        self.clear_box()
        pyautogui.write(offset_x)
        time.sleep(2)

        self.click_coordinate(offset_location[0]+offset_location[2]/2,vertical_location[1]+vertical_location[3]/2)
        self.clear_box()
        pyautogui.write(offset_y)
        time.sleep(2)

        self.click_coordinate(size_location[0]+size_location[2]/2,horizontal_location[1]+horizontal_location[3]/2)
        self.clear_box()
        pyautogui.write(size_x)
        time.sleep(2)

        self.click_coordinate(size_location[0]+size_location[2]/2,vertical_location[1]+vertical_location[3]/2)
        self.clear_box()
        pyautogui.write(size_y)
        time.sleep(2)

        self.click_coordinate(sampling_location[0]+sampling_location[2]/2,horizontal_location[1]+horizontal_location[3]/2)
        self.clear_box()
        pyautogui.write('128')
        time.sleep(2)

        self.click_coordinate(sampling_location[0]+sampling_location[2]/2,vertical_location[1]+vertical_location[3]/2)
        self.clear_box()
        pyautogui.write('128')

    def start_drprobe(self):
        #start drprobe
        pyautogui.press('win')
        pyautogui.write('dr')
        pyautogui.press('enter')

        # Wait for the application to open
        time.sleep(10)

    def clear_box(self):
        pyautogui.press('right',presses=50)
        pyautogui.press('backspace',presses=50)

    def wait_for_button(self,image,number_of_clicks=1,sleep_time=2):
        while(1):
            time.sleep(sleep_time)
            try:
                button_location = pyautogui.locateOnScreen(image, confidence=self.confidence_level)
                if button_location is not None:
                    button_point = pyautogui.center(button_location)
                    pyautogui.click(button_point,clicks=number_of_clicks)
                    #print(image+" has been found")
                    break
                #else:
                #    print(image+" has not been found")

            except pyautogui.ImageNotFoundException:
                #print(image+" has not been found")
                pass

    def sample_setup(self,file,sampling,num_slices):
        # Attempt to locate the button
        self.wait_for_button('ims/sample_setup.png')
        self.click('ims/sample_setup.png',self.confidence_level)
        self.default_mouse()
        time.sleep(2)

        self.wait_for_button('ims/select_file.png')
        self.default_mouse()
        time.sleep(2)

        pyautogui.write(file)
        pyautogui.press('enter')
        time.sleep(2)

        self.wait_for_button('ims/horizontally.png')
        horizontally_location = pyautogui.locateOnScreen("ims/horizontally.png", confidence=0.95)
        self.wait_for_button('ims/vertically.png')
        vertically_location = pyautogui.locateOnScreen("ims/vertically.png", confidence=0.95)
        self.wait_for_button('ims/discretization.png')
        discretization_location = pyautogui.locateOnScreen("ims/discretization.png", confidence=0.95)
        self.wait_for_button('ims/suggest.png',number_of_clicks=0)
        suggest_location = pyautogui.locateOnScreen("ims/suggest.png", confidence=0.95)
        self.wait_for_button('ims/variants_per_slice.png')
        variants_per_slice_location = pyautogui.locateOnScreen("ims/variants_per_slice.png", confidence=0.95)
        self.wait_for_button('ims/slices_along_z.png')
        slices_along_z_location = pyautogui.locateOnScreen("ims/slices_along_z.png", confidence=0.95)
        self.wait_for_button('ims/atomic_configurations.png',number_of_clicks=0)
        atomic_configurations_location = pyautogui.locateOnScreen("ims/atomic_configurations.png", confidence=0.95)

        self.click_coordinate(discretization_location[0]+discretization_location[2]/2,horizontally_location[1]+horizontally_location[3]/2)
        self.clear_box()
        pyautogui.write(sampling)
        self.default_mouse()
        time.sleep(2)

        self.click_coordinate(suggest_location[0]+suggest_location[2]/2,vertically_location[1]+vertically_location[3]/2)
        self.clear_box()
        pyautogui.write(sampling)
        self.default_mouse()
        time.sleep(2)

        self.click_coordinate(atomic_configurations_location[0]+atomic_configurations_location[2]/3,slices_along_z_location[1]+slices_along_z_location[3]/2)
        self.clear_box()
        pyautogui.write(str(num_slices))
        self.default_mouse()
        time.sleep(2)

        self.click_coordinate(suggest_location[0]+suggest_location[2]/2,variants_per_slice_location[1]+variants_per_slice_location[3]/2)
        self.clear_box()
        pyautogui.write('10')
        self.default_mouse()
        time.sleep(2)

        self.wait_for_button('ims/start_slice_creation.png')
        self.click('ims/start_slice_creation.png',self.confidence_level)
        self.default_mouse()
        time.sleep(2)

        self.wait_for_button('ims/slice_calculation_finished.png')
        self.click('ims/ok_slice_creation.png',self.confidence_level)
        self.default_mouse()
        time.sleep(2)

        self.wait_for_button('ims/OK.png')
        self.click('ims/OK.png',self.confidence_level)
        print("sample setup finished")
        time.sleep(2)

    def calculation_setup(self,offset_x="1.5",offset_y="1.5",size_x="3.9",size_y="3.9"):
        
        self.wait_for_button('ims/calculation_setup.png')
        self.click('ims/calculation_setup.png',self.confidence_level)
        self.default_mouse()
        time.sleep(2)

        self.wait_for_button('ims/scanning.png')
        self.click('ims/scanning.png',self.confidence_level)
        self.default_mouse()
        time.sleep(2)

        self.fill_calc_setup(offset_x,offset_y,size_x,size_y)
        time.sleep(2)

        self.wait_for_button('ims/ok_calc_setup.png')
        self.click('ims/ok_calc_setup.png',self.confidence_level)
        self.default_mouse()
        print("calculation setup finished")
        time.sleep(2)

    def run_calculation(self,i):
        
        self.wait_for_button('ims/start_calculation.png')
        self.click('ims/start_calculation.png',self.confidence_level)
        self.default_mouse()
        time.sleep(2)


        self.wait_for_button('ims/yes.png')
        self.click('ims/yes.png',self.confidence_level)
        self.default_mouse()
        time.sleep(2)
        
        self.wait_for_button('ims/calculation_finished.png',sleep_time=20)
        print("calculation finished")
        time.sleep(2)

    def save_data(self,data_name):
        self.wait_for_button('ims/save_results_to_file.png')
        self.click('ims/save_results_to_file.png',self.confidence_level)
        self.default_mouse()
        time.sleep(2)

        #implement "if no path"

        self.wait_for_button('ims/file_title.png')
        file_title_location = pyautogui.locateOnScreen("ims/file_title.png", confidence=0.9)
        self.wait_for_button('ims/current_det.png')
        curr_dect_location = pyautogui.locateOnScreen("ims/current_det.png", confidence=0.9)
        self.wait_for_button('ims/format.png')
        format_location = pyautogui.locateOnScreen("ims/format.png", confidence=0.9)
        #print("file title",file_title_location)
        #print("curr det",curr_dect_location)
        #print("format",format_location)
        
        pyautogui.moveTo(curr_dect_location[0]+curr_dect_location[2]/2,file_title_location[1]+file_title_location[3]/2)
        pyautogui.click()
        self.clear_box()
        pyautogui.write(data_name)
        time.sleep(2)


        for j in range(2):
            pyautogui.moveTo(curr_dect_location[0]+curr_dect_location[2]/2,format_location[1]+format_location[3]/2)
            pyautogui.click()
            self.reset_dropdown()
            if j == 0:
                pyautogui.press('down',presses=2) #MRC
            else:
                pyautogui.press('down',presses=5) #TIF
            pyautogui.press('enter')
            time.sleep(2)

            self.wait_for_button('ims/write_files.png')
            self.default_mouse()
            time.sleep(2)

        self.wait_for_button('ims/exit.png')
        self.default_mouse()
        time.sleep(2)

    def quit_drprobe(self):
        pyautogui.hotkey('alt', 'f4')
        pyautogui.press('enter')
        self.default_mouse()
        time.sleep(5)

    def generate_simulated_images(self):
        #start drprobe
        for i in range(self.dataset):
            if i != 1234:
                continue
            self.start_drprobe()
            #read structure_i with pkl
            with open(f"pkl/structure_{i}.pkl","rb") as f:
                dataframe = pkl.load(f)
            offset_x = np.round(dataframe["beam_offset_x"][0],2)
            #print("offset_x",offset_x)
            offset_y = np.round(dataframe["beam_offset_y"][0],2)
            pixel_size = dataframe["pixel_size"][0]
            size_x = np.round(pixel_size*128,2)
            size_y = np.round(pixel_size*128,2)
            k_max = 70
            slice_thickness = 0.1 #nm
            num_slices = np.floor((size_x+offset_x*2)/slice_thickness).astype(int)
            #print("num_slices",num_slices)
            #print("cell_side_length",size_x+offset_x*2)
            sampling = ((size_x+offset_x*2)*3*k_max).astype(int)
            #print("sampling",sampling) 
            print("pixel_size",pixel_size)
            self.sample_setup(f"{i}.cif",f"{sampling}",num_slices)
            self.calculation_setup(f"{offset_x}",f"{offset_y}",f"{size_x}",f"{size_y}")
            self.run_calculation(i)
            self.save_data(f"{i}")
            self.quit_drprobe()
            print(i)
#%%
if __name__ == "__main__":
    #run through all structures and print how many have pixel size smaller than 0.15
    j = 0
    k = 0
    lowest_pixel_size = 1
    for i in range(2000):
        print(i)
        if i == 1234:
            continue
        with open(f"pkl/structure_{i}.pkl","rb") as f:
            dataframe = pkl.load(f)
        pixel_size = dataframe["pixel_size"][0]
        surface_facet = dataframe["support_interface"][0]
        particle_surface_facet = dataframe["particle_interface"][0]
        if surface_facet == "random" and particle_surface_facet == "random":
            k += 1
        if pixel_size < 0.015:
            if pixel_size < lowest_pixel_size:
                lowest_pixel_size = pixel_size
            j += 1
    print(k)
    print(j)
    print(lowest_pixel_size)

    drprobe = DrProbeRunner()
    drprobe.generate_simulated_images()
# %%
#try:
#    while True:
#        x, y = pyautogui.position()
#        position_str = f'X: {x} Y: {y}'
#        print(position_str)
#        time.sleep(0.1)
#except KeyboardInterrupt:
#    print('\nDone.')
# %%
#import pickle as pkl
#import pandas as pd
#with open(f"pkl/structure_1.pkl","rb") as f:
#    dataframe = pkl.load(f)
#offset_x = dataframe["beam_offset_x"]
#print("offset_x",offset_x[0])
# %%
