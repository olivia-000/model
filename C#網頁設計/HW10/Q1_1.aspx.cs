using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;

public partial class _Default : System.Web.UI.Page
{

    protected void Page_Load(object sender, EventArgs e)
    {
        System.Web.UI.ValidationSettings.UnobtrusiveValidationMode =
        System.Web.UI.UnobtrusiveValidationMode.None;
    }

    protected void btnCalc_Click(object sender, EventArgs e)
    {
        if (Page.IsValid)
        {
            double h_cm = double.Parse(txtHeight.Text);
            double w = double.Parse(txtWeight.Text);
            double bmi = w / Math.Pow(h_cm / 100, 2);

            // 判斷體位
            string status = bmi < 18.5 ? "體重過輕"
                          : bmi <= 24 ? "健康體位"
                          : "體重過重";

            // 依 value 轉稱謂
            string title = ddlGender.SelectedValue == "1" ? "先生" : "小姐";

            string msg = $"{txtName.Text} {title}您好，您的 BMI 值 = {bmi:F2}，{status}。";

            // 存入 Session 並導向
            Session["Q1Msg"] = msg;
            Response.Redirect("Q1_2.aspx");
        }
    }
}