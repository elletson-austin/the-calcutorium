import logging
import sys

from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication

from .ui.main_window import MainWindow


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Depth buffer for 3D, 4x antialiasing. The default context on Windows is a
    # compatibility profile at the driver's highest version, which has the compute
    # shaders (OpenGL 4.3) the simulations need.
    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)

    qt_app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
