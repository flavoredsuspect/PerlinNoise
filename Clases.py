import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pt
import matplotlib.path as mpath
from shapely.geometry import MultiPolygon, Polygon, Point
from math import prod
import operator
import time
import progressbar
import tkinter as tk


class create_mask():

    def __init__(self,rot,*board):

        self.root = tk.Toplevel(rot)
        self.is_closed = False
        self.root.geometry("+300+100")

        self.board = np.where(np.ones((4, 6), dtype='str') == "1", "white", "white")

        if len(board) != 0:
            self.board = board[0]

        # In the code section below we define and place the menu elements: buttons for weight and height, text boxes...

        #   We start with a window of 4x6 pixels to fit well the default window, grid is for positioning and can't be used
        # in the variable value assignment because it will assign bult-in method

        tk.Label(self.root, text='H:').grid(row=0, column=0)
        tk.Label(self.root, text='W:').grid(row=1, column=0)

        self.pheight_btn = tk.Button(self.root, command=lambda: self.__change_dim__("pheight"), text='+1', fg="red",
                                     bd=1)
        self.pwidth_btn = tk.Button(self.root, command=lambda: self.__change_dim__("pwidth"), text='+1', fg="red",
                                    bd=1)
        self.pheight_btn.grid(row=0, column=1)
        self.pwidth_btn.grid(row=1, column=1)

        tk.Label(self.root, text='4', fg='red').grid(row=0, column=2)
        tk.Label(self.root, text='6', fg='red').grid(row=1, column=2)

        self.mheight_btn = tk.Button(self.root, command=lambda: self.__change_dim__("mheight"), text='-1', fg="red",
                                     bd=1)
        self.mwidth_btn = tk.Button(self.root, command=lambda: self.__change_dim__("mwidth"), text='-1', fg="red",
                                    bd=1)
        self.mheight_btn.grid(row=0, column=3)
        self.mwidth_btn.grid(row=1, column=3)

        self.height_txt = tk.Text(self.root, width=2, height=1)
        self.width_txt = tk.Text(self.root, width=2, height=1)
        self.height_txt.grid(row=0, column=4)
        self.width_txt.grid(row=1, column=4)

        tk.Button(self.root, text='set', command=lambda: self.__take_input__("height"), width=2, height=1,
                  bd=1).grid(row=0, column=5)
        tk.Button(self.root, text='set', command=lambda: self.__take_input__("width"), width=2, height=1,
                  bd=1).grid(row=1, column=5)

        self.print_board()
        self.root.protocol('WM_DELETE_WINDOW', self.__on_close__)
        self.root.mainloop()

    def __on_close__(self):
        self.is_closed = True
        self.root.destroy()
        self.root.quit()

    def __take_input__(self, dir):

        #   This function takes care of the expanding and decreasing dimensions of board with text-box values, this is
        # acquired by increasing or decreasing the board as many times as the difference between the current board and
        # the input difference indicates. This process is slow again because of the print_board function that shpuld
        # become refreshing function.

        if dir == "height":
            inpt = self.height_txt.get("1.0", tk.END)

            if inpt.strip().isdigit():
                for i in range(0, abs(self.board.shape[0] - int(inpt))):
                    if self.board.shape[0] - int(inpt) < 0:
                        self.__change_dim__("pheight")
                    elif self.board.shape[0] - int(inpt) > 0:
                        self.__change_dim__("mheight")

        elif dir == "width":
            inpt = self.width_txt.get("1.0", tk.END)
            if inpt.strip().isdigit():
                for i in range(0, abs(self.board.shape[1] - int(inpt))):
                    if self.board.shape[1] - int(inpt) < 0:
                        self.__change_dim__("pwidth")
                    elif self.board.shape[1] - int(inpt) > 0:
                        self.__change_dim__("mwidth")

    def __on_click__(self, i, j, event):

        #   This function takes care of setting the black adn white color to board, and is binded in print_board to the left-mouse
        # button, it's work is pretty simple. Are the binding cummulative? In this case white and black discriminant is essential.

        if self.board[i][j] == "white":
            self.board[i][j] = "black"
        elif self.board[i][j] == "black":
            self.board[i][j] = "white"
        event.widget.config(bg=self.board[i][j])

    def print_board(self):

        #   This function prints the labels of the actual mask in white and binds them to the left-mouse click within the function on_click
        # is really uneffective because it prints and binds the board each time from 0, could be improved without difficulties.
        # the i rows are shifted by 3: 2 rows for the upper-menu and 1 row for margin between menu and board, noting that when
        # passing the rows and cols to the on_click function this change needs to be reverted because we are using the board as an object
        # not as a part of the current window root. One more essential change to be done is not summing up bindings in window, this should
        # be done within the refreshing.

        for k in range(0, self.board.shape[1]):
            tk.Label(self.root, text='', bg=self.root.cget('bg')).grid(row=2, column=k)

        for i in range(0, self.board.shape[0]):
            i += 3
            for j in range(0, self.board.shape[1]):
                L = tk.Label(self.root, text='      ', bg=str(self.board[i-3,j]))
                L.grid(row=i, column=j)
                L.bind('<Button-1>', lambda e, i=i, j=j: self.__on_click__(i - 3, j, e))

    def __erase_board__(self):

        #   This function sets values of bg to grey when you decrease the dimensions, the binding is not erased, to be fixed
        # with saving board of labels, and in place of changing color, deleting.

        for i in range(0, self.board.shape[0]):
            i += 3
            for j in range(0, self.board.shape[1]):
                L = tk.Label(self.root, text='      ', bg=self.root.cget('bg'))
                L.grid(row=i, column=j)

    def __change_dim__(self, action):

        #   This function increases and decreases board's dimensions, by appending or deleting current board, erasing
        #    the previous one off the window root and then reprinting.

        if action == "pheight":
            app = np.where(np.ones((1, self.board.shape[1]), dtype='str') == "1", "white", "white")
            self.board = np.append(self.board, app, axis=0)
            self.print_board()

        elif action == "mheight":
            if self.board.shape[0] > 1:
                self.__erase_board__()
                self.board = np.delete(self.board, self.board.shape[0] - 1, 0)
                self.print_board()

        elif action == "pwidth":
            app = np.where(np.ones((self.board.shape[0], 1), dtype='str') == "1", "white", "white")
            self.board = np.append(self.board, app, axis=1)
            self.print_board()

        elif action == "mwidth":
            if self.board.shape[1] > 1:
                self.__erase_board__()
                self.board = np.delete(self.board, self.board.shape[1] - 1, 1)
                self.print_board()

        tk.Label(self.root, text=str(self.board.shape[0]), fg='red').grid(row=0, column=2)
        tk.Label(self.root, text=str(self.board.shape[1]), fg='red').grid(row=1, column=2)
        # self.root.geometry = str(self.board.shape[0]) + 'x' + str(self.board.shape[1] + self.menu_dim)

class Menu():
    # The main Menu() class manages the mask creation to ease handling multiple masks.

    def __init__(self):
        self.root= tk.Tk()
        self.masks_btn= list()
        self.masks=list()
        self.counter=0
        self.current=0
        self.opened_window=False

        self.exit= tk.Button(self.root,text="EXIT",width=6,height=3, fg='red', command= lambda: self.__terminate__())
        self.exit.grid(row=0,column=0)

        self.counter_lbl= tk.Label(self.root,text=str(self.counter),height=3,width=4)
        self.counter_lbl.grid(row=0,column=1)

        self.join_btn= tk.Button(self.root,text="+",width=6,height=3,wraplength=17,command= lambda: self.__join__())
        self.join_btn.grid(row=1,column=0)

        self.root.protocol('WM_DELETE_WINDOW', self.__terminate__)

        self.root.mainloop()

    def __join__(self):

        # This function takes care of adding buttons of current windowns to update the counters
        #and creating buttons when a new window for mask creating is requested, ONLY ONE mask can
        #be created at a time, this is intentially done because the tkinter library does not manage
        # easly several windows due to the event nature and mainloops.

        if not self.opened_window:
            self.counter += 1
            self.current += 1
            self.counter_lbl.configure(text=str(self.counter))
            self.join_btn.grid(row=self.counter+1,column=0)

            self.opened_window = True
            a=create_mask(self.root)
            self.opened_window = False

            self.masks.append(a.board)
            self.masks_btn.append(tk.Button(self.root, text=str(self.counter), width=6, height=2,command=lambda: self.__show_mask__(len(self.masks))))
            self.masks_btn[-1].grid(row=self.counter, column=0)

    def __show_mask__(self,index):

        # This function calls create_mask() and allows to desing one shape, as mentioned, while
        #this process is running the rest of the main Menu() is frozen.

        if not self.opened_window:
            self.opened_window = True
            a=create_mask(self.root,self.masks[index-1])
            self.opened_window = False
            self.masks[index-1]=a.board

    def __terminate__(self):

        # As the name suggests the function handles the exit process when the custom exit button is used
        #and also whren the default red cross of the corener of main root is clicked.

        self.close=True
        self.opened_window=True
        self.vis= tk.Toplevel()
        txt= tk.Label(self.vis,text="Are you sure you want to exit?")
        txt.pack(side=tk.TOP)
        tk.Button(self.vis,text="Yes",command= lambda: self.__but__(True)).pack(side=tk.BOTTOM)
        tk.Button(self.vis, text="No", command=lambda: self.__but__(False)).pack(side=tk.BOTTOM)

        self.vis.mainloop()

        if self.close:
            self.root.quit()
            self.root.destroy()

    def __but__(self,a):

        # Method to call when the custom exit button is clicked, the .quit() and .destroy()
        #methods are called in this order to ensure the corrent functioning of tkinter roots.

        self.opened_window=False
        self.close= a
        self.vis.quit()
        self.vis.destroy()

class Data():

    def __init__(self,image,model):
        self.image=image                                                             # The image to be transformed, needs to have correct model dimensions
        self.model=model                                                             # The model to cheat
        self.prediction= model.predict(self.image.reshape(1, 28, 28, 1)).argmax()    # The prediction for the current image of the model, to achieve missclasification
        self.frequencies = np.zeros(self.image.shape)                                # Number of times an occupied pixel was in a misclassification, this praises changed regions more that a whole image
        self.count=0                                                                 # Number of times a pixel was occupied and checked, to divide frequencies achieving range [0,1]
        self.configurations = list()
        self.configurations.append(np.ones(self.image.shape))                        # List of current non-updated boards, we start the list with 1 matrix board all empty
        self.depths=list()                                                           # List of current depth, each depth matches the same index as the configuration that holds it
        self.depths.append(0)
        self.progress=progressbar.ProgressBar(max_value=progressbar.UnknownLength).start() # Current bar of progress indicates depth, we change the third default widgte.
        self.progress.widgets[3]=' Curreent Depth '

    def __delta__(self,A, mask):

        # This funcion checks if a given polygon in matrix form fits inside a pice of an image, this is achieved by reshaping
        # the pice of image A and the mask (polygon matrix) into a vector, multiply them and perform modulus operator:
        #
        #       2-Value defines an occupied space
        #       3-Value defines a free space
        #
        # Because in the nZ group with n=4 [2]*[2]=[0], [3]*[3]=[2], [2]*[3]=[1] we achieve this relations:
        #
        #       Free board space[3] * Free mask space[3]= [1]
        #       Free board space[3] * Non-Free mask space[2] = [2] = Non-Free board space[2] * Free mask space[3]
        #       Non-Free board space[2] * Non-Free space[2] = [0]
        #
        # The result of multiplying all values is 0 if we have a collision.

        A = np.where(A == 0, 2, 3)
        mask = np.where(mask == 0, 2, 3)

        # We convert the 0-occupied and 1-free space to 2 and 3, respectively

        if A.shape == mask.shape:
            R = A.reshape(A.shape[0] * A.shape[1]) * mask.reshape(mask.shape[0] * mask.shape[1])
            R = R.astype(int)
            R %= 4
            if not R.prod() == 0:
                R %= 2
                # We have [1] values in occupied space and [2] values in free space at this point so we have to go back to
                # standart [1]-free [0]-non-free, we sum 1 and the conversion is completed
                return R.reshape(A.shape)
            else:
                return np.ones(A.shape) * 3
        else:
            print(
                "Los tamaños de la máscara y la submatriz no son iguales")  # Warning that the mask and submatrix are needed to have same shape

    def __update__(self,A, show):

        # This function updates the frequencies of the current A board evaluated in image with the model

        # First we combine the image with the board, multiplying and setting occupied values to 0 (black color), then
        # we evaluate the model and change values 0 to 1 and viceversa for summing frequencies. We do the same
        # to calculate the number of times a pixel was occupied and evaluated, no matter the result.

        b=[np.array_equal(x,A) for x in self.configurations]
        B=self.configurations.pop(b.index(True))
        self.depths.pop(b.index(True))
        pred= (255-self.image.reshape(prod(self.image.shape)))*A.reshape(prod(A.shape))
        pred=255-pred
        pred=pred.reshape((A.shape))
        result=self.model.predict(pred.reshape(1,28,28,1)).argmax()

        if result!=self.prediction:
            self.frequencies += np.where(B==0,1,0)

        self.count += np.where(B==0,1,0)

        if show==True:

            # This last section is to visualize the regions blacked in image, the value for dimensions is related
            # to dpi but i didn't used it i divided by 5 to avhieve a window greater that image pxls

            fig=plt.figure(figsize=tuple(x/5 for x in self.image.shape))
            ax=fig.add_subplot()
            im=ax.imshow(pred, cmap='Greys')
            ax.text(self.image.shape[0],self.image.shape[1],str(result)+" "+str(self.prediction),fontsize='medium')
            plt.pause(3)
            input()
            plt.close(fig)

    def __checkout__(self, A, mask,show):

        #This function performs the task of checking the possible combinations of a mask inside a configuration

        # forbidden is a configuration of the image with polygons or shapes already settled, in terms of data, is a matrix,
        # filled with 0 and 1 that represents an abstract shape where 0 are shape's body and 1 is free space, same as the mask variable
        # representing abstract polygon

        b = [np.array_equal(x, A) for x in self.configurations]
        ValidP=list()

        # ValidP is a list of configurations that can be included in the actual image extendign the forbidden configuration
        # The b array is to find de index of current config in configurations list because .index method seems not to work

        for i in range(0,A.shape[0]-mask.shape[0]+1):
            for j in range(0,A.shape[1]-mask.shape[1]+1):
                B=np.copy(A[i:i+mask.shape[0],j:j+mask.shape[1]])
                Z=np.copy(A)
                Z[i:i+mask.shape[0],j:j+mask.shape[1]]=self.__delta__(B,mask)

                if not 3 in Z:
                    ValidP.append(Z)
                    self.depths.append(self.depths[b.index(True)]+1)

        # In this section we check for every rectangle with the same dimensions of the mask the delta function, wich by
        # modulus operations proves if the actual mask fits in the submatrix of the board and returns the valid polygons
        # The depths are updated for valid configurations and increased by one of the parent

        self.__update__(A,show)
        self.configurations += ValidP

        #In this sections we update the old board and concatenate to the non-updated lists the ValidP
        # REMEMBER in the last board to update won't use this method

    def spoil(self,depth,masks,show):

        # In this function the full shape-matching into the image is done.

        # depth: is the maximum nested depth that is allowed to process, it increases extremely fast when the mask
        #        has low dimensions. Depth 0 is considered as the one evaluation of the basic ones-board into the frequencies.
        # masks: is a list of masks that are going to be used in each index-matching depth step, if only one is provided, it
        #        will be used in all steps.
        # show:  this variable is boolean and asks for the all-states visualizing

        if len(masks)==1:

            d=masks[0].copy()

            for i in range(0,depth):
                masks.append(d)

        # This cycle takes care of the case whrn only one mask is given

        # In the next block we use recursion and call spoil for the first term of configurations list, then calculates
        #   all possible matches thanks to checking and finally updates the stats and pops-out the first term. When we
        #   achieve the desired depth one last update is done to accomplish real statistics.

        if len(self.depths)!=0:

            self.progress.update(self.depths[0])

            if self.depths[0]<depth:
                self.__checkout__(self.configurations[0],masks[self.depths[0]],show)

            if self.depths[0]==depth:
                self.__update__(self.configurations[0],show)

            return(self.spoil(depth, masks, show))
        else:
            self.progress.finish()
            return self.frequencies/np.where(self.count==0, 1, self.count)

        # In this last code lines, we pass return twice, because we are dealing with nested functions and we need some
        #   return argument, each nested step demands the same function to return variable and only in the last step,
        #   when the list of configuratios is empty, we then return the data frequencies. The last line is to ensure
        #   0-safe division and 0-1 range of frequencies.

## bar plot and analysis
## improve show, real time-state bar plots etc
