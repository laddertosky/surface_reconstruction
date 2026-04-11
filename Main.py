import open3d.visualization.gui as gui  # type: ignore

from Layout import Window


def main():
    app = gui.Application.instance
    app.initialize()
    Window(app=app, name="Surface Reconstruction", width=1920, height=720)
    app.run()

if __name__ == "__main__":
    main()
