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

   * View (`PySideRenderSpace` class):
       * Visually represents the Scene (the Model).
       * It's a QOpenGLWidget responsible for all moderngl rendering.
       * It handles all input that happens directly within the 3D viewport, such as camera controls
         (panning, rotating, zooming) and 3D object selection/interaction (via ray casting).       

   * Controller (`MainWindow` class):
       * The central orchestrator of the application.
       * It creates and owns all major components: the Scene (Model), the PySideRenderSpace (View),
         and all other GUI elements (buttons, sliders, input fields).
       * It connects the View to the Model (e.g., view.set_scene(model)).
       * It handles all application-level user interaction. For example, when a "Plot" button is
         clicked, a method in the MainWindow is called. This method then updates the Scene by
         adding a new function object to it. The View then automatically redraws the updated Scene.

  This separation of concerns allows for rapid prototyping. Adding new features primarily involves
  adding a UI element to the MainWindow and writing a corresponding method to modify the Scene,
  without needing to touch the complex rendering code in the View.

TODO improving parsing (dont want np.sin(x) want sin(x)).
TODO add functions in terminal style widget.
TODO add desmos style grid that is slightly see through in 3d but opaque in 2d
TODO allow the setting of x0,x1,y0,y1 manually while in 2d to change aspect ratio.
TODO add more particle sims
TODO add polar and implicit function compatibility
TODO add the ability to graph functions like f(x,y) = some z val and it looks like a mesh
TODO add way to tell f(x)'s apart. (f-sub1, fsub2) using some kind of subscript. or allow custom names for functions.
TODO when in 2d mode it takes the ranges and evaluates the function so it doesnt abruptly stop. (updates vertices)
