Hi Gemini. 
My name is Austin and I am in highschool going into college.
this folder contains a project that is important for me and my mathematical journey
its going to be my calculator, similar to desmos but larger in scale.
everything lives in a 3d space but if I need 2d I change the perspective.
Id like to use pyside6 for a gui implementation.
its important to me that I know the tool inside and out so I don't want you to generate massive amounts of the project unless I explicitly ask for that.
I want to take my time and implement it right. start sloppy and iterate relentlessly.
Project Architecture: A Model-View-Controller (MVC) Approach

  This project uses a Model-View-Controller (MVC) architectural pattern to ensure code is
  organized, scalable, and easy to iterate on.

   * Model (`Scene` class):
       * Represents the application's data, acting as the single source of truth.
       * It holds the list of all 3D objects (axes, functions, etc.).
       * It is "dumb" and does not contain any rendering or user interaction logic. It only manages
         the scene data.

   * View (`RenderSpace` class):
       * Visually represents the Scene (the Model).
       * It's a QOpenGLWidget responsible for all moderngl rendering.
       * It handles all input that happens directly within the 3D viewport, such as camera controls
         (panning, rotating, zooming) and 3D object selection/interaction (via ray casting).       

   * Controller (`MainWindow` class):
       * The central orchestrator of the application.
       * It creates and owns all major components: the Scene (Model), the RenderSpace (View),
         and all other GUI elements (buttons, sliders, input fields).
       * It connects the View to the Model (e.g., view.set_scene(model)).
       * It handles all application-level user interaction. For example, when a "Plot" button is
         clicked, a method in the MainWindow is called. This method then updates the Scene by
         adding a new function object to it. The View then automatically redraws the updated Scene.

  This separation of concerns allows for rapid prototyping. Adding new features primarily involves
  adding a UI element to the MainWindow and writing a corresponding method to modify the Scene,
  without needing to touch the complex rendering code in the View.

  Dont paint yourself into a corner. keep everything as applical to the widest array of uses.
  Have an extremely agnostic and unopinionated approach
  
TODO add which variables are in the domain and the output.

TODO add more particle sims
TODO add polar and implicit function compatibility
TODO add the ability to graph functions like f(x,y) = some z val and it looks like a mesh
TODO Generalize MathFunction plotting for different 2D planes.
    - Currently, `MathFunction` only plots `y = f(x)` on the XY plane.
    - Extend this to plot functions with a single independent variable (e.g., `f(x)`, `f(y)`) on any 2D plane where that variable is an axis.
    - For example, `f(x)` should appear on the XY and XZ planes, but not the YZ plane.
    - This involves:
        1. In `scene.py`, modifying `MathFunction` to auto-detect its independent variable.
        2. Modifying its vertex generation to place points correctly based on the current 2D plane (`xy`, `xz`, `yz`).
        3. In `rendering.py`, modifying `RenderSpace.paintGL` to pass the plane info and the correct range to the `MathFunction` object.
