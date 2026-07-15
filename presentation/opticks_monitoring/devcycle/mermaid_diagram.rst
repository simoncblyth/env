mermaid_diagram
================

Overview

* see workflow image in ~/simoncblyth.github.io/env/presentation/opticks_monitoring/devcycle/



gemma4 query::

    Please generate some ascii art that illustrates the Opticks+JUNOSW development
    cycle. I want a circle connecting boxes for the stages:  1. geometry changes,
    physics model changes 2. optimization 3. added features (eg server-client,
    non-NVIDIA GPU) 4. validation by comparison with Geant4 5. performance
    monitoring



Not much good result::


                    [  1. Geometry & Physics  ]
                   /                           \
                  v                             |
        [  5. Performance Mon.  ]        [ 2. Optimization ]
                 ^                             |
                 |                             v
         [ 4. Geant4 Validation ] <--- [  3. Added Features  ]
                (Comparison)           (GPU, Server/Client)

    -----------------------------------------------------------
                          LEGEND:
          1. Geometry changes / Physics model changes
          2. Optimization phase
          3. New Features (e.g., server-client, non-NVIDIA GPU)
          4. Validation via comparison with Geant4
          5. Performance Monitoring & Analysis



                 +--------------------------+
                 |  1. GEOMETRY / PHYSICS   |
                 |        CHANGES           |
                 +------------+-------------+
                              |
                              v
                 +------------+-------------+       +----------------------+
                 |      2. OPTIMIZATION      |----->|  3. ADDED FEATURES   |
                 +------------+-------------+       | (GPU, Server-Client) |
                              ^                     +----------+-----------+
                              |                                |
                              |                                v
                 +------------+-------------+        +----------------------+
                 |  5. PERFORMANCE MON.      | <----- | 4. GEANT4 VALIDATION |
                 |                           |        |      (Comparison)    |
                 +--------------------------+        +----------------------+





Are you able to generate an image ? That illustrates this development cycle ?

I cannot directly "draw" a pixel-based image file (like a .jpg or .png) myself,
but I can do something much better for technical documentation: I can write the
code for a professional-grade diagram.

The Professional Diagram (Mermaid.js) Most modern documentation tools (GitHub,
Notion, Obsidian, and many Markdown editors) will render this code into a
beautiful, crisp, high-resolution flowchart automatically. It is much more
readable than ASCII or AI-generated art.


::

    graph TD
        subgraph "Opticks+JUNOSW Development Cycle"
        A[<b>1. Geometry & Physics</b><br/>Changes to models/geometry] --> B(<b>2. Optimization</b><br/>Code efficiency improvements)
        B --> C(<b>3. Feature Expansion</b><br/>Server-Client, non-NVIDIA GPU)
        C --> D(<b>4. Validation</b><br/>Comparison with Geant4)
        D --> E(<b>5. Performance Monitoring</b><br/>System health & speed)
        E --> A
        end

        %% Styling to make it look professional
        style A fill:#f9f,stroke:#333,stroke-width:2px
        style B fill:#bbf,stroke:#333,stroke-width:2px
        style C fill:#bfb,stroke:#333,stroke-width:2px
        style D fill:#fbb,stroke:#333,stroke-width:2px
        style E fill:#dff,stroke:#333,stroke-width:2px


Copy pasting that into https://mermaid.live/ yields an OK - but very wide diagram.

I used Mermiad live, it looks OK - but the result is too wide - how can I influence the rendering ?



Mermaid's layout engine (which uses a tool called dagre) tries to find the most
"logical" path, which often results in very wide horizontal layouts if your
text is long or if you are using graph LR (Left-to-Right).

To make it narrower and more vertical, you can use these three specific techniques:

1. Change the Direction (TD instead of LR)
The most important command is at the very top of your code:

graph LR: Left $\rightarrow$ Right (Very wide).
graph TD: Top $\downarrow$ Down (Narrower/Vertical).
graph BT: Bottom $\uparrow$ Top.
2. Force Text Wrapping with <br/>
This is the most effective way to fix a "wide" diagram. By default, Mermaid
tries to put all your text on one single line, which pushes the boxes far
apart. You must manually insert HTML break tags (<br/>) inside the labels to
force the text to stack vertically.

Example:

Wide: [Geometry changes and physics model changes]
Narrow: [Geometry changes<br/>and physics<br/>model changes]
3. Use "Invisible" Links (Advanced)
If you want two boxes that are side-by-side to actually sit on top of each
other to save width, you can use a "hidden" connection, but usually, simply
switching to graph TD is enough.


I have rewritten your specific content using the <br/> technique and graph TD
direction. Copy and paste this into Mermaid Live::

    graph TD
        %% Using TD makes it vertical/narrow
        %% Using <br/> inside brackets prevents horizontal stretching

        A["<b>1. Geometry & Physics</b><br/>(Model Changes)"] 
        --> B["<b>2. Optimization</b><br/>(Code Efficiency)"]
        
        B --> C["<b>3. Added Features</b><br/>(Server-Client,<br/>non-NVIDIA GPU)"]
        
        C --> D["<b>4. Validation</b><br/>(Comparison with<br/>Geant4)"]
        
        D --> E["<b>5. Performance Mon.</b><br/>(System Metrics)"]
        
        E --> A

        %% Styling to make it look clean
        style A fill:#f9f,stroke:#333,stroke-width:2px
        style B fill:#bbf,stroke:#333,stroke-width:2px
        style C fill:#bfb,stroke:#333,stroke-width:2px
        style D fill:#fbb,stroke:#333,stroke-width:2px
        style E fill:#dff,stroke:#333,stroke-width:2px



Summary of Changes made to the code above:

graph TD: Changed from LR to ensure it flows downwards.

<br/> tags: I added these inside every single box (e.g., non-NVIDIA GPU became
non-NVIDIA<br/>GPU). This "squeezes" the boxes, making them taller rather than
wider.

Quotations " ": When you use HTML tags like <br/> or <b> inside a node, you
must wrap the label in double quotes (e.g., A["Text<br/>Text"]) otherwise the
code will crash.

HTML Bold <b>: I added bolding to the titles so they stand out from the
descriptive text, which helps with readability when the diagram is narrow.






















graph TD
    subgraph "Opticks+JUNOSW Development Cycle"
    A[<b>1. Geometry & Physics</b><br/>Changes to models/geometry] --> B(<b>2. Optimization</b><br/>Code efficiency improvements)
    B --> C(<b>3. Feature Expansion</b><br/>Server-Client, non-NVIDIA GPU)
    C --> D(<b>4. Validation</b><br/>Comparison with Geant4)
    D --> E(<b>5. Performance Monitoring</b><br/>System health & speed)
    E --> A
    end

    %% Styling to make it look professional
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#bfb,stroke:#333,stroke-width:2px
    style D fill:#fbb,stroke:#333,stroke-width:2px
    style E fill:#dff,stroke:#333,stroke-width:2px

