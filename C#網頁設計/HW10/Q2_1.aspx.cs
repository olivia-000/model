using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;

public partial class Q2_1 : System.Web.UI.Page
{
    protected void Page_Load(object sender, EventArgs e)
    {
        if (!IsPostBack)
        {
            for (int i = 1; i <= 10; i++)
                ddlYears.Items.Add(i.ToString());
            ddlYears.Items.Insert(0, new ListItem("--請選擇--", ""));
        }
    }

    protected void btnCalc_Click(object sender, EventArgs e)
    {
        if (Page.IsValid)
        {
            // 將資料存入 Cookie，設定 1 小時後失效
            HttpCookie ck = new HttpCookie("Q2");
            ck["Name"] = txtName.Text;
            ck["Principal"] = txtPrincipal.Text;
            ck["Rate"] = txtRate.Text;
            ck["Years"] = ddlYears.SelectedValue;
            ck.Expires = DateTime.Now.AddHours(1);
            Response.Cookies.Add(ck);

            Response.Redirect("Q2_2.aspx");
        }
    }
}