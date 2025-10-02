using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;

public partial class Q3_2 : System.Web.UI.Page
{
    protected void Page_Load(object sender, EventArgs e)
    {

    }

    protected void btnNext_Click(object sender, EventArgs e)
    {
        if (Page.IsValid)
        {
            Session["Addr"] = txtAddr.Text;
            Session["Tel"] = txtTel.Text;
            Session["Birth"] = txtBirth.Text;
            Response.Redirect("Q3_3.aspx");
        }
    }
}