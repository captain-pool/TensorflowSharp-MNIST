using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using TensorFlow;
namespace TF_Test
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }
        private TFTensor img_tensor = null;
        private void button1_Click(object sender, EventArgs e)
        {
            //int val1=Int32.Parse(textBox1.Text);
            //int val2 = Int32.Parse(textBox2.Text);
            //var sess = new TFSession();
            //var tf = sess.Graph;
            //var a = tf.Const(val1);
            //var b = tf.Const(val2);
            //label1.Text = sess.GetRunner().Run(tf.Add(a, b)).GetValue().ToString();
            //sess.CloseSession();
            using (var graph = new TFGraph())
            {
                var file = File.ReadAllBytes(@"..\..\python\tmp\model\output.pb");
                graph.Import(file);

                var tf = new TFSession(graph);
                var runner = tf.GetRunner();
                if (img_tensor != null)
                {
                    runner.AddInput(graph["input"][0], img_tensor);
                    runner.Fetch(graph["output"][0]);
                    var output = runner.Run();
                    TFTensor result = output[0];


                    label1.Text ="Prediction:\n"+ ((System.Int64[])result.GetValue())[0].ToString();
                }
                
                
            }
                
            


        }

        private void button2_Click(object sender, EventArgs e)
        {
            openFileDialog1.DefaultExt = "jpg";
            openFileDialog1.CheckFileExists = true;
            openFileDialog1.FileName = "";
            openFileDialog1.CheckPathExists = true;
            //openFileDialog1.InitialDirectory = @"C:\";
            openFileDialog1.Filter = "JPG Files (*.jpg)|*.JPG|PNG Files(*.png)|*.PNG";
            var dr = openFileDialog1.ShowDialog();
            if(dr == DialogResult.OK)
            {
                img_tensor = null;
                Bitmap image = new Bitmap(openFileDialog1.FileName);
                if (image.Height != 28 && image.Width != 28)
                {
                    MessageBox.Show("Provide a Photo of Size 28x28!", "Warning");
                    return;
                }
                pictureBox1.Image = image;
                pictureBox1.SizeMode = PictureBoxSizeMode.StretchImage;
                var image_tensor = ImageUtility.FormatJPEG(openFileDialog1.FileName);
                img_tensor = image_tensor;

            }
        }
    }
}
