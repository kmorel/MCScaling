#include <MCScalingConfig.h>

#include <stdlib.h>
#include <stdio.h>

#include <fstream>
#include <iostream>

#include <vector>

#define DAX_DEVICE_ADAPTER DAX_DEVICE_ADAPTER_ERROR

#include <dax/CellTag.h>
#include <dax/CellTraits.h>
#include <dax/cont/ArrayContainerControlBasic.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/GenerateTopology.h>
#include <dax/cont/Scheduler.h>
#include <dax/cont/Timer.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>

#include <dax/worklet/Magnitude.h>
#include <dax/worklet/Threshold.h>

#include <dax/tbb/cont/DeviceAdapterTBB.h>

#include <tbb/task_scheduler_init.h>

#define LOAD_DATA 1
//#define REMOVE_UNUSED_POINTS 1

class CopyWorklet : public dax::exec::WorkletMapField
{
public:
  typedef void ControlSignature(Field(In), Field(Out));
  typedef _2 ExecutionSignature(_1);

  template<typename T>
  DAX_EXEC_EXPORT T operator()(const T &inValue) const
  {
    return inValue;
  }
};

class ThresholdExample
{
private:

  typedef dax::cont::ArrayContainerControlTagBasic Container;
  typedef dax::tbb::cont::DeviceAdapterTagTBB DeviceAdapter;

  typedef dax::cont::UniformGrid<DeviceAdapter> UniformGridType;
  typedef dax::cont::UnstructuredGrid<
      dax::CellTagHexahedron,Container,Container,DeviceAdapter>
      UnstructuredGridType;

  typedef dax::cont::ArrayHandle<
      dax::Scalar, dax::cont::ArrayContainerControlTagBasic, DeviceAdapter>
      ArrayHandleScalar;
  typedef dax::cont::ArrayHandle<
      dax::Vector3, dax::cont::ArrayContainerControlTagBasic, DeviceAdapter>
      ArrayHandleVector;

  static UniformGridType CreateUniformGrid()
  {
    UniformGridType grid;
    grid.SetExtent(dax::make_Id3(0, 0, 0),
                   dax::make_Id3(GRID_SIZE-1, GRID_SIZE-1, GRID_SIZE-1));
    return grid;
  }

  static void ReadSupernovaData(std::vector<dax::Scalar> &buffer)
  {
    assert(sizeof(float) == sizeof(dax::Scalar));

    FILE *fd = fopen(DATA_FILE, "rb");
    assert(fd != NULL);

    buffer.resize(GRID_SIZE*GRID_SIZE*GRID_SIZE);
    fread(&buffer.front(), sizeof(float), GRID_SIZE*GRID_SIZE*GRID_SIZE, fd);
    assert(ferror(fd) == 0);

    fclose(fd);
  }

  static void CreateMagnitudeField(const UniformGridType &grid,
                                   ArrayHandleScalar &data)
  {
    dax::cont::Scheduler<DeviceAdapter> scheduler;
    scheduler.Invoke(dax::worklet::Magnitude(),
                     grid.GetPointCoordinates(),
                     data);
  }

  static void WriteGradientMagnitude(const ArrayHandleScalar &data)
  {
    FILE *fd = fopen("/Users/kmorel/Downloads/gradients.dat", "wb");
    assert(fd != NULL);

    typedef ArrayHandleScalar::PortalConstControl PortalType;
    PortalType dataPortal = data.GetPortalConstControl();

    for (PortalType::IteratorType iter = dataPortal.GetIteratorBegin();
         iter != dataPortal.GetIteratorEnd();
         iter++)
      {
      dax::Scalar value = *iter;
      fwrite(&value, sizeof(dax::Scalar), 1, fd);
      }

    fclose(fd);
  }

  template<typename Stream>
  static void PrintContentsToStream(UnstructuredGridType& grid, Stream &stream)
  {
    const dax::Id num_points(grid.GetNumberOfPoints());
    const dax::Id num_cells(grid.GetNumberOfCells());

    //dump header
    stream << "# vtk DataFile Version 3.0"  << std::endl;
    stream << "vtk output" << std::endl;
    stream << "ASCII"  << std::endl;
    stream << "DATASET UNSTRUCTURED_GRID" << std::endl;
    stream << "POINTS " << num_points << " float" << std::endl;

    std::vector<dax::Vector3> contPoints(num_points);
    grid.GetPointCoordinates().CopyInto(contPoints.begin());

    for(dax::Id i=0; i < num_points; ++i)
      {
      dax::Vector3 coord = contPoints[i];
      stream << coord[0] << " " << coord[1] << " " << coord[2] << " ";
      if(i%4==3)
        {
        stream << std::endl; //pump new line after each 4th vector
        }
      }
    if(num_points%4==3)
      {
      stream << std::endl;
      }

    const int NUM_VERTICES =
        dax::CellTraits<dax::CellTagHexahedron>::NUM_VERTICES;
    //print cells
    stream << "CELLS " << num_cells << " " << num_cells * (NUM_VERTICES+1)
           << std::endl;

    std::vector<dax::Id> contTopo(num_cells*NUM_VERTICES);
    grid.GetCellConnections().CopyInto(contTopo.begin());

    dax::Id index=0;
    for(dax::Id i=0; i < num_cells; ++i,index+=8)
      {
      stream << NUM_VERTICES << " ";
      stream << contTopo[index+0] << " ";
      stream << contTopo[index+1] << " ";
      stream << contTopo[index+2] << " ";
      stream << contTopo[index+3] << " ";
      stream << contTopo[index+4] << " ";
      stream << contTopo[index+5] << " ";
      stream << contTopo[index+6] << " ";
      stream << contTopo[index+7] << " ";

      stream << std::endl;
      }
    stream << std::endl;
    stream << "CELL_TYPES " << num_cells << std::endl;
    for(dax::Id i=0; i < num_cells; ++i)
      {
      stream << "12" << std::endl; //11 is voxel && 12 is hexa
      }
  }

public:
  static int Run()
  {
    FILE *fd = fopen("MCScalingTbb.csv", "w");
    assert(fd != NULL);
    fprintf(fd, "Implementation,Threads,Trial,Seconds\n");

    try
    {
#ifdef LOAD_DATA
    std::cout << "Reading data..." << std::endl;
    std::vector<dax::Scalar> buffer;
    ReadSupernovaData(buffer);
    std::cout << "Data read." << std::endl;
#endif

    UniformGridType grid = CreateUniformGrid();

    int maxNumThreads = tbb::task_scheduler_init::default_num_threads();
    std::cout << "Max num threads: " << maxNumThreads << std::endl;

    for (int numThreads = 1; numThreads <= maxNumThreads; numThreads++)
      {
      tbb::task_scheduler_init tbbSchedulerInit(numThreads);
      for (int trial = 0; trial < NUM_TRIALS; trial++)
        {
        dax::cont::Scheduler<DeviceAdapter> scheduler;

#ifdef LOAD_DATA
        ArrayHandleScalar originalArray =
            dax::cont::make_ArrayHandle(buffer, Container(), DeviceAdapter());
        ArrayHandleScalar inArray;
        std::cout << "Copying data for memory affinity." << std::endl;
        scheduler.Invoke(CopyWorklet(), originalArray, inArray);
#else // LOAD_DATA
        std::cout << "Creating data" << std::endl;
        ArrayHandleScalar inArray;
        CreateMagnitudeField(grid, inArray);
#endif // LOAD_DATA
        assert(grid.GetNumberOfPoints() == inArray.GetNumberOfValues());

//        std::cout << "Computing gradient..." << std::endl;
//        dax::cont::ArrayHandle<dax::Vector3> gradient;
//        dax::cont::worklet::CellGradient(
//              grid, grid.GetPointCoordinates(), inArray, gradient);
//        inArray.ReleaseResources();

//        std::cout << "Computing magnitude..." << std::endl;
//        dax::cont::ArrayHandle<dax::Scalar> magnitude;
//        dax::cont::worklet::Magnitude(gradient, magnitude);
//        gradient.ReleaseResources();

//        std::cout << "Writing gradient magnitude..." << std::endl;
//        WriteGradientMagnitude(magnitude);
//        std::cout << "Data written." << std::endl;

        std::cout << "Computing Threshold, " << numThreads << " threads..."
                  << std::endl;
        dax::cont::Timer<DeviceAdapter> timer;
        typedef dax::cont::ArrayHandle<dax::Id, Container, DeviceAdapter>
            ClassifyResultType;
        typedef dax::cont::GenerateTopology<
            dax::worklet::ThresholdTopology,ClassifyResultType>
            GenerateTopologyType;

        // Run classify algorithm (determine how many cells are passed).
#ifdef LOAD_DATA
        const dax::Scalar LOW_SCALAR = 0.07;
        const dax::Scalar HIGH_SCALAR = 1.0;
#else // LOAD_DATA
        const dax::Scalar LOW_SCALAR = 50.0;
        const dax::Scalar HIGH_SCALAR = 200.0;
#endif // LOAD_DATA
        ClassifyResultType classificationArray;
        scheduler.Invoke(dax::worklet::ThresholdClassify<dax::Scalar>(
                           LOW_SCALAR, HIGH_SCALAR),
                         grid,
                         inArray,
                         classificationArray);

        // Build thresholded topology.
        GenerateTopologyType resolveTopology(classificationArray);
#ifndef REMOVE_UNUSED_POINTS
        resolveTopology.SetRemoveDuplicatePoints(false);
#endif
        UnstructuredGridType outGrid;
        scheduler.Invoke(resolveTopology, grid, outGrid);

#ifdef REMOVE_UNUSED_POINTS
        // Compact scalar array to new topology.
        ArrayHandleScalar outArray;
        resolveTopology.CompactPointField(inArray, outArray);

        // Copy grid information to host, if necessary.
        outGrid.GetCellConnections().GetPortalConstControl();
        outGrid.GetPointCoordinates().GetPortalConstControl();
        outArray.GetPortalConstControl();
#endif

        dax::Scalar time = timer.GetElapsedTime();
        std::cout << "Number of output cells: " << outGrid.GetNumberOfCells()
                  << std::endl;
        std::cout << "Time: " << time << " seconds" << std::endl;
        fprintf(fd, "Dax-TBB,%d,%d,%lg\n",numThreads,trial,time);
//        std::cout << "CSV," << description << "," << time << "," << trial << std::endl;
        }
      }

//    std::cout << "Writing threshold geometry..." << std::endl;
//    std::ofstream file;
//    file.open("/Users/kmorel/Downloads/thresholdGeometry.vtk");
//    PrintContentsToStream(outGrid, file);
//    file.close();
//    std::cout << "Done." << std::endl;
    }
    catch (dax::cont::Error error)
    {
      std::cout << "Caught Dax error: " << std::endl
                << error.GetMessage() << std::endl;
      return 1;
    }

    fclose(fd);

    return 0;
  }
};

int main(int, char *[])
{
  return ThresholdExample::Run();
}

