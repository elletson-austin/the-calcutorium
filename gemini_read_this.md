Hi Gemini. 
My name is Austin.
this gonna be my calculator, similar to desmos but larger in scale (not in polish).
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
  # Important below
  The expression itself isn't a string to be parsed; it's a structured tree of
  mathematical objects that you build and edit directly. This is a much more powerful and intuitive concept, similar to how modern calculator       
  interfaces work.
  The core idea is that we are not typing code; we are visually building an Abstract Syntax Tree (AST)
  
  The goal is to have the AST's responsibility end with the
  creation of a valid mathematical string. The MathFunction object will only know how to parse that string with SymPy, keeping the two domains
  cleanly separated.

  ---

  1. The Model: The Abstract Syntax Tree (AST)

  This is the most critical part. Every expression will be represented in memory as a tree of nodes.

   * `Node` (Base class): The fundamental building block.
   * `ConstantNode(Node)`: Represents a number (e.g., 5, 3.14).
   * `VariableNode(Node)`: Represents a variable (e.g., x, y).
   * `BinaryOpNode(Node)`: Represents an operation with two children, like +, -, *, /, ^. For x + 2, the BinaryOpNode(+) would have a
     VariableNode(x) and a ConstantNode(2) as its children.
   * `UnaryOpNode(Node)`: For operations with one child, like a negative sign.
   * `FunctionNode(Node)`: Represents a function call, like sin(x). It would hold the function name (sin) and have one or more children representing
     the arguments.
   * `EquationNode(Node)`: The root of the tree for a full equation, holding the left-hand side and right-hand side as children.
   * `PlaceholderNode(Node)`: A special, temporary node that represents an empty, editable space—the "blinking cursor" of the editor.

  Why this is key: This tree is the expression. It's unambiguous and can be directly evaluated, differentiated, or manipulated without any string
  parsing.

  ---

  2. The View: Visual Rendering of the AST

  This component's job is to display the AST in a human-readable, graphical way.

   * No `QLineEdit`. Instead, each expression in the list will be a custom-drawn QWidget.
   * This widget will render the AST. It will walk the tree and draw its contents. For a BinaryOpNode('^') (power), it would draw the first child,
     then draw the second child as a smaller superscript. For a BinaryOpNode('/') (divide), it would draw a horizontal line and render the children
     above and below it.
   * It needs to visually indicate the current cursor position within the tree. For example, a blinking cursor in an empty PlaceholderNode, or a
     highlight around an existing node that can be replaced.
   * Layout Engine: The View is more than just a painter; it's a layout engine. It must recursively calculate the bounding box of every node before
     drawing. To draw (x+1)/(y+1), it must first calculate the size of x+1 and y+1 to determine how long the fraction bar needs to be and how to
     center the numerator and denominator.
   * Hit Testing: The View must be able to translate a mouse click back into a selection. When you click on the screen, the View will traverse its
     layout information to find which specific node in the AST corresponds to that coordinate.

  ---

  3. The Controller: The Keypad and Editing Logic

  This is how the user interacts with and modifies the AST. It connects the keypad/keyboard to the data model (the AST).

   * A "Keypad" Widget: This is a separate panel with buttons for numbers (0-9), operators (+, -, ^), functions (sin, cos, log), and structural     
     elements (fractions, parentheses).
   * Cursor & Selection Management:
       * The "cursor" is simply a reference to the currently active Node in the AST (often a PlaceholderNode).
       * Arrow keys will allow the user to navigate the tree: up/down might move between numerator/denominator, while left/right moves between      
         siblings (e.g., from x to 2 in x+2).
   * Keyboard & Keypad Input Handling:
       1. Input is received (e.g., user clicks the 'sin' button or types 's' on the keyboard).
       2. The Controller identifies the currently selected Node (the cursor).
       3. It modifies the tree directly. If a PlaceholderNode is selected and the user clicks sin, the controller replaces that placeholder with a  
          FunctionNode('sin') that itself contains a new PlaceholderNode for its argument.
       4. The Controller then signals the View to update, which re-renders the visual expression.
   * Structural Editing: The Controller also handles higher-level edits. For example, a user could select an entire sub-tree (like x+2), and then   
     press the ( key. The Controller would wrap the selected nodes within a new ParenthesesNode, preserving the internal structure.

  This approach is far more robust and is the standard for building modern symbolic editors. It completely avoids the ambiguity and complexity of   
  parsing strings.

TODO add more particle sims
TODO add polar and implicit function compatibility
TODO add the ability to graph functions like f(x,y) = some z val and it looks like a mesh

