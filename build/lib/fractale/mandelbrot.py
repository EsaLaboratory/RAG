import argparse
from fractale.fractal import plot_mandelbrot

def main():
    """Création de la commande shell MandelbrotPlot."""
    parser = argparse.ArgumentParser(description='Affiche la fractalle de Mandelbrot')
    parser.add_argument('--zmin', metavar='zmin', type=str,
                        help="coordonnées coin inférieur gauche de l'image")
    parser.add_argument('--zmax', metavar='zmax', type=str,
                        help="coordonnées du coin supérieur droit de l'image")
    parser.add_argument('--pixel_size', metavar='pixel_size', type=str,
                        help="taile de pixel de l'image")
    parser.add_argument('--max_iter', metavar='max_iter', type=str,
                        help="nombre d'itérations pour le test de divergeance")
    parser.add_argument('--o', metavar='filename', type=str,
                        help='titre de la figure')

    args = parser.parse_args()

    zmin = complex(args.zmin) if args.zmin is not None else -2-2*complex(0,1)
    zmax = complex(args.zmax) if args.zmax is not None else 2+2*complex(0,1)
    pixel_size = float(args.pixel_size) if args.pixel_size is not None else float(5e-4)
    max_iter = int(args.max_iter) if args.pixel_size is not None else int(50)
    filename = args.o if args.o is not None else "Mandelbrot"

    plot_mandelbrot(zmin=zmin, zmax=zmax, pixel_size=pixel_size, max_iter=max_iter, filename=filename)

if __name__=="__main__":
    main()