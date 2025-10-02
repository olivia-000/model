using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;

public partial class Q3_3 : System.Web.UI.Page
{
    protected void Page_Load(object sender, EventArgs e)
    {

    }

    protected void btnShow_Click(object sender, EventArgs e)
    {
        if (Page.IsValid)
        {
            Session["Hobby"] = ddlHobby.SelectedValue;

            lblAll.Text = $@"
            帳號：{Session["User"]}<br/>
            密碼：{Session["Pwd"]}<br/>
            地址：{Session["Addr"]}<br/>
            電話：{Session["Tel"]}<br/>
            生日：{Session["Birth"]}<br/>
            興趣：{Session["Hobby"]}";
        }
    }
}