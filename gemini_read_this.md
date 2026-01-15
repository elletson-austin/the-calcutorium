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

TODO Parsing improvements
TODO in 2d space change so mouse drag is proportional to the space. (like holding down a point in space and dragging yourself off of it)
TODO add more particle sims
TODO add polar and implicit function compatibilityc
TODO add the ability to graph functions like f(x,y) = some z val and it looks like a mesh.
TODO add way to tell f(x)'s apart. (f-sub1, fsub2) using some kind of subscript. or allow custom names for functions.
TODO when in 2d mode it takes the ranges and evaluates the function so it doesnt abruptly stop. (updates vertices)
TODO add some kind of eval command that you can pick the function and evaluate it at some value. (implement after functions have custom titles)

## Gemini TODOs
- **TODO:** Implement picking/selection of objects in the 3D scene. This would allow users to interact with plotted functions or other objects directly.
- **TODO:** Add support for more complex function types, such as parametric equations (e.g., `x(t)`, `y(t)`, `z(t)`).
- **TODO:** Enhance the `InputWidget` to have a history of commands (e.g., using up/down arrow keys).
- **TODO:** Implement a file dialog for the `save` and `load` commands, instead of requiring the user to type the filename.
- **TODO:** Add more detailed error messages to the `OutputWidget` to help with debugging.
- **TODO:** Create a more comprehensive user manual or help system.
- **TODO:** Performance optimization for rendering a large number of points or complex functions.
- **TODO:** Add UI elements for controlling Lorenz attractor parameters (sigma, rho, beta).