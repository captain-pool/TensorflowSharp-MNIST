using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;
using System.IO;
namespace TF_Test
{
    class ImageUtility
    {
        public static TFGraph ImConvertGraph(TFTensor image, out TFOutput input,out TFOutput output)
        {
            var graph = new TFGraph();
            input = graph.Placeholder(TFDataType.String);
            const int W = 28;
            const int H = 28;
            output = graph.Div(graph.ResizeBilinear(graph.ExpandDims(graph.Cast(
                graph.DecodePng(contents: input, channels: 1), DstT: TFDataType.Float),
                dim: graph.Const(0)),size: graph.Const(new int[] {W,H})
                ),y:graph.Const((float)255)
                );
            return graph;
        }
        public static TFTensor FormatJPEG(string filePath)
        {
            var d = File.ReadAllBytes(filePath);
            var g = TFTensor.CreateString(d);
            TFOutput input, output;
            using (var graph = ImConvertGraph(g, out input, out output))
            {
                using (var session = new TFSession(graph))
                {
                    var runner = session.GetRunner();
                    runner.AddInput(input, g);
                    var r = runner.Run(output);
                    return r;
                }
            }
        }
    }
}
