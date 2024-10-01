import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import random
import time
import sys

# overall entry point
def main(argv):
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()

# takes a filename, reads a sudoku in and returns a string
def readSudokuFromFile(infile):
    try:
        with open(infile, 'r') as f:
            initpuzzle = [line.split() for line in f]
    except IOError as err:
        print(f"<<{err}>>")
        with open("./puzzle1.txt", 'r') as f:
            initpuzzle = [line.split() for line in f]
    return initpuzzle

# separate (from the gui) thread to do the actual PSO processing
class PSOThread(threading.Thread):
    def __init__(self, parent=None):
        threading.Thread.__init__(self)
        self.pause = True
        self.finished = False
        self.infile = "./puzzle1.txt"
        self.nparticles = 150
        self.weights = [.9, .4, .5]
        self.lazythreshold = 5
        self.timestamp = 0
        self.parent = parent

    # run() is the entry point of the thread, contains our main processing loop
    def run(self):
        random.seed()
        initpuzzle = readSudokuFromFile(self.infile)
        s = swarm(initpuzzle, self.nparticles, self.weights, self.lazythreshold)
        self.parent.updateSwarm(s)
        self.parent.initPso()

        while not self.finished:
            if not self.pause:
                s.optimize()
                self.parent.updatePso()
                timestamp = time.time()
                self.parent.timePso(timestamp - self.timestamp)
                self.timestamp = timestamp
            else:
                time.sleep(0.5)
        self.parent.endPso()

    def setInfile(self, infile):
        self.infile = infile

    def setWeights(self, w):
        self.weights = w

    def setNparticles(self, p):
        self.nparticles = p

    def setLazyThreshold(self, l):
        self.lazythreshold = l

    def pausePso(self):
        self.pause = not self.pause

    def resetPso(self):
        self.finished = not self.finished

# main window for our gui
class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("PSO Sudoku Solver")
        self.currentParticle = 0
        self.p = PSOThread(self)

        self.particlemodel = ParticleModel(self)
        self.particleview = ttk.Treeview(root, columns=("Particle", "Fitness", "PBest"), show='headings')
        self.particleview.heading("Particle", text="Particle")
        self.particleview.heading("Fitness", text="Fitness")
        self.particleview.heading("PBest", text="PBest")
        self.particleview.pack(side=tk.LEFT, fill=tk.Y)

        self.puzzlemodel = PuzzleModel(self)
        self.puzzleview = ttk.Treeview(root, columns=[str(i) for i in range(9)], show='headings')
        for i in range(9):
            self.puzzleview.heading(str(i), text=str(i))
        self.puzzleview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.velmodel = PuzzleModel(self)
        self.velview = ttk.Treeview(root, columns=[str(i) for i in range(9)], show='headings')
        for i in range(9):
            self.velview.heading(str(i), text=str(i))
        self.velview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.pbestmodel = PuzzleModel(self)
        self.pbestview = ttk.Treeview(root, columns=[str(i) for i in range(9)], show='headings')
        for i in range(9):
            self.pbestview.heading(str(i), text=str(i))
        self.pbestview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.gbestmodel = PuzzleModel(self)
        self.gbestview = ttk.Treeview(root, columns=[str(i) for i in range(9)], show='headings')
        for i in range(9):
            self.gbestview.heading(str(i), text=str(i))
        self.gbestview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.puzzlelabel = tk.Label(root, text="Puzzle 0")
        self.puzzlelabel.pack()
        self.vellabel = tk.Label(root, text="Velocity 0")
        self.vellabel.pack()
        self.pbestlabel = tk.Label(root, text="Personal Best")
        self.pbestlabel.pack()
        self.gbestlabel = tk.Label(root, text="Global Best")
        self.gbestlabel.pack()
        self.lazythreshlabel = tk.Label(root, text="Laziness Threshold")
        self.lazythreshlabel.pack()
        self.nparticleslabel = tk.Label(root, text="Number of Particles")
        self.nparticleslabel.pack()
        self.velweightlabel = tk.Label(root, text="Velocity Weight")
        self.velweightlabel.pack()
        self.personalweightlabel = tk.Label(root, text="Personal Best Weight")
        self.personalweightlabel.pack()
        self.globalweightlabel = tk.Label(root, text="Global Best Weight")
        self.globalweightlabel.pack()
        self.filelabel = tk.Label(root, text="Input File")
        self.filelabel.pack()
        self.timelabel = tk.Label(root, text="Elapsed Time")
        self.timelabel.pack()
        self.particletimelabel = tk.Label(root, text="Elapsed Time per Particle")
        self.particletimelabel.pack()

        self.lazythreshedit = tk.Entry(root)
        self.lazythreshedit.insert(0, str(self.p.lazythreshold))
        self.lazythreshedit.pack()
        self.nparticlesedit = tk.Entry(root)
        self.nparticlesedit.insert(0, str(self.p.nparticles))
        self.nparticlesedit.pack()
        self.velweightedit = tk.Entry(root)
        self.velweightedit.insert(0, str(self.p.weights[0]))
        self.velweightedit.pack()
        self.personalweightedit = tk.Entry(root)
        self.personalweightedit.insert(0, str(self.p.weights[1]))
        self.personalweightedit.pack()
        self.globalweightedit = tk.Entry(root)
        self.globalweightedit.insert(0, str(self.p.weights[2]))
        self.globalweightedit.pack()
        self.fileedit = tk.Entry(root)
        self.fileedit.insert(0, str(self.p.infile))
        self.fileedit.pack()

        self.resetbutton = tk.Button(root, text="Reset", command=self.resetPso)
        self.resetbutton.pack()
        self.pausebutton = tk.Button(root, text="Run", command=self.pausePso)
        self.pausebutton.pack()

        self.puzzlecheck = tk.Checkbutton(root, text="Decimal", command=self.toggleDecimal)
        self.puzzlecheck.pack()
        self.velcheck = tk.Checkbutton(root, text="Decimal", command=self.toggleDecimal)
        self.velcheck.pack()
        self.pbestcheck = tk.Checkbutton(root, text="Decimal", command=self.toggleDecimal)
        self.pbestcheck.pack()
        self.gbestcheck = tk.Checkbutton(root, text="Decimal", command=self.toggleDecimal)
        self.gbestcheck.pack()

        self.p.start()

    def changeParticle(self, p):
        if isinstance(p, int):
            self.currentParticle = p
        else:
            self.currentParticle = p.row()
        self.puzzlelabel.config(text=f"Puzzle {self.currentParticle}")
        self.vellabel.config(text=f"Velocity {self.currentParticle}")
        self.particlemodel.setParticle(self.currentParticle)
        self.puzzlemodel.setValid(self.swarm.particles[self.currentParticle].valid)
        self.puzzlemodel.setData(self.swarm.particles[self.currentParticle].sudoku.puzzle)
        self.velmodel.setValid(self.swarm.particles[self.currentParticle].valid)
        self.velmodel.setData(self.swarm.particles[self.currentParticle].velocity)
        self.pbestlabel.config(text=f"Personal Best (Particle {self.currentParticle}: {self.swarm.particles[self.currentParticle].personalbest})")
        self.pbestmodel.setValid(self.swarm.particles[self.currentParticle].personalbestvalid)
        self.pbestmodel.setData(self.swarm.particles[self.currentParticle].personalbestposition)
        self.gbestlabel.config(text=f"Global Best (Particle {self.swarm.globalbestparticle}: {self.swarm.globalbest})")
        self.gbestmodel.setValid(self.swarm.globalbestvalid)
        self.gbestmodel.setData(self.swarm.globalbestposition)

    def updateSwarm(self, s):
        self.swarm = s

    def timePso(self, elapsed):
        self.timelabel.config(text=f"Elapsed Time: {round(elapsed, 5)}s")
        self.particletimelabel.config(text=f"Elapsed Time: {round(elapsed / self.swarm.nparticles, 5)}s/particle")

    def resetPso(self):
        self.p.finished = True
        self.p.join()
        self.p = PSOThread(self)
        self.p.setInfile(self.fileedit.get())
        self.p.setNparticles(int(self.nparticlesedit.get()))
        self.p.setWeights([float(self.velweightedit.get()), float(self.personalweightedit.get()), float(self.globalweightedit.get())])
        self.p.setLazyThreshold(int(self.lazythreshedit.get()))
        self.p.start()

    def pausePso(self):
        self.p.pausePso()
        if self.p.pause:
            self.pausebutton.config(text="Run", bg="green")
        else:
            self.pausebutton.config(text="Pause", bg="red")

    def initPso(self):
        self.particlemodel.setData(self.swarm)
        self.puzzlemodel.setMask(self.swarm.particles[0].sudoku.puzzlemask)
        self.puzzlemodel.setValid(self.swarm.particles[self.currentParticle].valid)
        self.puzzlemodel.setData(self.swarm.particles[self.currentParticle].sudoku.puzzle)
        self.velmodel.setMask(self.swarm.particles[0].sudoku.puzzlemask)
        self.velmodel.setValid(self.swarm.particles[self.currentParticle].valid)
        self.velmodel.setData(self.swarm.particles[self.currentParticle].velocity)
        self.pbestmodel.setMask(self.swarm.particles[0].sudoku.puzzlemask)
        self.pbestmodel.setValid(self.swarm.particles[0].personalbestvalid)
        self.pbestmodel.setData(self.swarm.particles[0].personalbestposition)
        self.gbestmodel.setMask(self.swarm.particles[0].sudoku.puzzlemask)
        self.gbestmodel.setValid(self.swarm.globalbestvalid)
        self.gbestmodel.setData(self.swarm.globalbestposition)

    def updatePso(self):
        self.puzzlemodel.setValid(self.swarm.particles[self.currentParticle].valid)
        self.velmodel.setValid(self.swarm.particles[self.currentParticle].valid)
        self.pbestlabel.config(text=f"Personal Best (Particle {self.currentParticle}: {self.swarm.particles[self.currentParticle].personalbest})")
        self.pbestmodel.setValid(self.swarm.particles[self.currentParticle].personalbestvalid)
        self.pbestmodel.setData(self.swarm.particles[self.currentParticle].personalbestposition)
        self.gbestlabel.config(text=f"Global Best (Particle {self.swarm.globalbestparticle}: {self.swarm.globalbest})")
        self.gbestmodel.setValid(self.swarm.globalbestvalid)
        self.gbestmodel.setData(self.swarm.globalbestposition)

    def toggleDecimal(self):
        self.puzzlemodel.setDecimal(self.puzzlecheck.var.get())
        self.velmodel.setDecimal(self.velcheck.var.get())
        self.pbestmodel.setDecimal(self.pbestcheck.var.get())
        self.gbestmodel.setDecimal(self.gbestcheck.var.get())

class ParticleModel:
    def __init__(self, parent=None):
        self.arraydata = [[None, None, None]]
        self.currentParticle = 0

    def setData(self, data):
        self.arraydata = data

    def setParticle(self, p):
        self.currentParticle = p

class PuzzleModel:
    def __init__(self, parent=None):
        self.arraydata = [[0 for j in range(9)] for i in range(9)]
        self.validdata = ["none"]
        self.puzzlemask = [[0 for j in range(9)] for i in range(9)]
        self.outputdecimal = False

    def setDecimal(self, decimal):
        self.outputdecimal = decimal

    def setValid(self, valid):
        self.validdata = valid

    def setData(self, data):
        self.arraydata = data

    def setMask(self, mask):
        self.puzzlemask = mask

if __name__ == "__main__":
    main(sys.argv)