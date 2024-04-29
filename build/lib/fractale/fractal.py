import numpy as np
import matplotlib.pyplot as plt
import numba

j = complex(0, 1)

@numba.jit(nopython=True)
def suite(z,c)-> complex:
    """Générateur des éléments de la suite $z_{n+1}=z_n^2+c$
    
    c.f. Chapitre 2"""
    while True:
        yield z
        z = z ** 2 + c


@numba.jit(nopython=True) # On optimise les calculs avec numba
def suite_mandelbrot(z:complex, c:complex)-> complex:
    """Renvoie un générateur des termes de la suite de Mandelbrot considérée.

    :param z: Valeur initiale z_0 de la suite.
    :type z: complex
    :param c: Paramètre c de la suite.
    :type c: complex
    :return: Un générateur des termes de la suite.
    :rtype: complex
    """
    return suite(z,c)


@numba.jit(nopython=True) # On optimise les calculs avec numba
def suite_julia(z:complex, c:complex)-> complex:
    """Renvoie un générateur des termes de la suite de Julia considérée.

    :param z: Valeur initiale z_0 de la suite.
    :type z: complex
    :param c: Paramètre c de la suite.
    :type c: complex
    :return: Un générateur des termes de la suite.
    :rtype: complex
    """
    return suite(z,c)


@numba.jit(nopython=True) # On optimise les calculs avec numba
def is_in_Mandelbrot(c:complex, max_iter:int=50):
    """Détermine si un complexe appartient à l'ensemble de Mandelbrot.

    :param c: Nombre complexe dont on vérifie l'appartenance à l'ensemble de Mandelbrot.
    :type c: complex
    :param max_iter: Nombre d'itérations après lequelles on considère que la suite converge et donc que le complexe appartient à l'ensemble, defaults to 50.
    :type max_iter: int, optional
    :return: Le complexe appartient ou non à l'ensemble.
    :rtype: boolean
    """

    if abs(c) > 2: # Les éléments de cet ensemble sont compris dans le cercle complexe de rayon 2
        return False 
    
    iterator = suite_mandelbrot(z=0, c=c)
    iter = 0 # On compte le nombre d'itérations
    for item in iterator:
        if abs(item) > 1e6: # On considère que la suite diverge
            return False
        if iter >= max_iter: # On considère que la suite n'a pas divergé pour les max_iter premiers termes donc elle converge
            return True
        iter += 1


@numba.jit(nopython=True) # On optimise les calculs avec numba
def is_in_Julia(z:complex, c:complex, max_iter:int=50):
    """Détermine si un complexe appartient à l'ensemble de Julia.

    :param z: Nombre complexe dont on vérifie l'appartenance à l'ensemble de Julia.
    :type z: complex
    :param c: Paramètre fixé de la suite.
    :type c: complex
    :param max_iter: Nombre d'itérations après lequelles on considère que la suite converge et donc que le complexe appartient à l'ensemble, defaults to 50.
    :type max_iter: int, optional
    :raises Exception: Si c=0 alors la suite n'est plus une suite de Julia mais de Mandelbrot.
    :return: Le complexe appartient ou non à l'ensemble.
    :rtype: boolean
    """
    if z==0:
        raise Exception("z must be different from 0") # Une suite de Julia avec z = 0 est une suite de Mandelbrot
    iterator = suite_julia(z=z, c=c)
    iter = 0# On compte le nombre d'itérations
    for item in iterator:
        if abs(item) > 1e6: # On considère que la suite diverge
            return False
        if iter >= max_iter: # On considère que la suite n'a pas divergé pour les max_iter premiers termes donc elle converge
            return True
        iter += 1


@numba.jit(nopython=True)
def compute_mandelbrot(zmin:complex=-2-2j, zmax:complex=2+2j, pixel_size:float=0.001, max_iter:int=100):
    """Calcule l'array correspondant à l'image de l'ensemble de Mandelbrot dans la fenêtre considérée.
    Le calcul est fait séparément de l'affichage avec pyplot pour pouvoir utiliser numba.

    :param zmin: Point en bas à gauche de la fenêtre des points considérés, defaults to -2-2j.
    :type zmin: complex, optional
    :param zmax: Point en haut à droite de la fenêtre des points considérés, defaults to 2+2j.
    :type zmax: complex, optional
    :param pixel_size: Pas de la grille de points dont l'apartenance à l'ensemble de Mandelbrot est testé, defaults to 0.01.
    :type pixel_size: float, optional
    :param max_iter: Nombre d'itérations après lequelles on considère que la suite converge et donc que le complexe appartient à l'ensemble, defaults to 50.
    :type max_iter: int, optional
    :return: L'array des points de l'image en noir et blanc.
    :rtype: np.array
    """
    X = np.linspace(np.real(zmin), np.real(zmax), num=int(1/pixel_size)) # Partie réelle
    Y = np.linspace(np.imag(zmax), np.imag(zmin), num=int(1/pixel_size)) # Partie imaginaire
    to_plot = np.empty((int(1/pixel_size), int(1/pixel_size))) # Image vide au départ

    for p in range(len(X)):
        for q in range(len(Y)):
            if is_in_Mandelbrot(c=X[p]+Y[q]*j, max_iter=max_iter):
                to_plot[q, p] = 1
    return to_plot


def plot_mandelbrot(zmin:complex=-2-2j, zmax:complex=2+2j, pixel_size:float=0.001, max_iter:int=100, filename:str="Mandelbrot"):
    """Affiche dans la fenêtre considérée l'ensemble de Mandelbrot obtenu par compute_mandelbrot.

    :param zmin: Point en bas à gauche de la fenêtre des points considérés, defaults to -2-2j.
    :type zmin: complex, optional
    :param zmax: Point en haut à droite de la fenêtre des points considérés, defaults to 2+2j.
    :type zmax: complex, optional
    :param pixel_size: Pas de la grille de points dont l'apartenance à l'ensemble de Mandelbrot est testé, defaults to 0.01.
    :type pixel_size: float, optional
    :param max_iter: Nombre d'itérations après lequelles on considère que la suite converge et donc que le complexe appartient à l'ensemble, defaults to 50.
    :type max_iter: int, optional
    :param filename: Nom du fichier où l'image du résultat est enregistrée, defaults to "Mandelbrot".
    :type filename: str, optional
    """
    to_plot = compute_mandelbrot(zmin=zmin, zmax=zmax, pixel_size=pixel_size, max_iter=max_iter)

    fig, ax = plt.subplots()
    ax.imshow(to_plot, cmap='Greys')
    ax.set_title(filename + f" {zmin} - {zmax}.png")
    ax.set_axis_off()
    plt.savefig(filename)
    plt.show()


@numba.jit(nopython=True)
def compute_julia(c:complex, zmin:complex=-2-2j, zmax:complex=2+2j, pixel_size:float=0.001, max_iter:int=100):
    """Calcule l'array correspondant à l'image de l'ensemble de Julia dans la fenêtre considérée.
    Le calcul est fait séparément de l'affichage avec pyplot pour pouvoir utiliser numba.

    :param c: Paramètre de la suite de Julia. 
    :type c: complex
    :param zmin: Point en bas à gauche de la fenêtre des points considérés, defaults to -2-2j.
    :type zmin: complex, optional
    :param zmax: Point en haut à droite de la fenêtre des points considérés, defaults to 2+2j.
    :type zmax: complex, optional
    :param pixel_size: Pas de la grille de points dont l'apartenance à l'ensemble de Julia est testé, defaults to 0.01.
    :type pixel_size: float, optional
    :param max_iter: Nombre d'itérations après lequelles on considère que la suite converge et donc que le complexe appartient à l'ensemble, defaults to 50.
    :type max_iter: int, optional
    :return: L'array des points de l'image en noir et blanc.
    :rtype: np.array
    """
    X = np.linspace(np.real(zmin), np.real(zmax), num=int(1/pixel_size)) # Partie réelle
    Y = np.linspace(np.imag(zmax), np.imag(zmin), num=int(1/pixel_size)) # Partie imaginaire
    to_plot = np.empty((int(1/pixel_size), int(1/pixel_size))) # Image vide au départ

    for p in range(len(X)):
        for q in range(len(Y)):
            if is_in_Julia(z=X[p]+Y[q]*j, c=c, max_iter=max_iter):
                to_plot[q, p] = 1
    return to_plot

def plot_julia(c:complex, zmin:complex=-2-2j, zmax:complex=2+2j, pixel_size:float=0.001, max_iter:int=100, filename:str="Julia"):
    """Affiche dans la fenêtre considérée l'ensemble de Julia obtenu par compute_julia.
    
    :param c: Paramètre de la suite de Julia. 
    :type c: complex
    :param zmin: Point en bas à gauche de la fenêtre des points considérés, defaults to -2-2j.
    :type zmin: complex, optional
    :param zmax: Point en haut à droite de la fenêtre des points considérés, defaults to 2+2j.
    :type zmax: complex, optional
    :param pixel_size: Pas de la grille de points dont l'apartenance à l'ensemble de Mandelbrot est testé, defaults to 0.01.
    :type pixel_size: float, optional
    :param max_iter: Nombre d'itérations après lequelles on considère que la suite converge et donc que le complexe appartient à l'ensemble, defaults to 50.
    :type max_iter: int, optional
    :param filename: Nom du fichier où l'image du résultat est enregistrée, defaults to "Julia".
    :type filename: str, optional
    """ 
    to_plot = compute_julia(c, zmin=zmin, zmax=zmax, pixel_size=pixel_size, max_iter=max_iter)

    fig, ax = plt.subplots()
    ax.imshow(to_plot, cmap='Greys')
    ax.set_title(filename + f" {zmin} - {zmax}.png")
    ax.set_axis_off()
    plt.savefig(filename)
    plt.show()