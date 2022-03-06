"""
https://gitlab.kitware.com/vtk/vtk/-/issues/17917#note_783584

"""
import vtk

arrowSource = vtk.vtkArrowSource()

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(arrowSource.GetOutputPort())
actor = vtk.vtkActor()
actor.SetMapper(mapper)

renderer = vtk.vtkRenderer()
renderer.AddActor(actor)

renderWindow = vtk.vtkRenderWindow()
renderWindow.SetWindowName("Arrow")
renderWindow.AddRenderer(renderer)

for i in range(300):
  renderer.GetActiveCamera().Azimuth(1)
  renderWindow.Render()

#renderWindow.Finalize()
#del renderWindow

