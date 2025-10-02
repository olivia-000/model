using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;

public partial class Q1_2 : System.Web.UI.Page
{
    protected void Page_Load(object sender, EventArgs e)
    {
        lblResult.Text = Session["Q1Msg"]?.ToString() ?? "Session 遺失，請重新計算。";
    }

}