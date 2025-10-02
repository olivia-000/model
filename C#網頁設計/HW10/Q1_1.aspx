<%@ Page Language="C#" AutoEventWireup="true" CodeFile="Q1_1.aspx.cs" Inherits="_Default" %>

<!DOCTYPE html>

<html xmlns="http://www.w3.org/1999/xhtml">
<head runat="server">
<meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
    <title></title>
</head>
<body>
    <form id="form1" runat="server">
        <div>
        姓名：<asp:TextBox ID="txtName" runat="server" /><br />
        性別：
        <asp:DropDownList ID="ddlGender" runat="server">
            <asp:ListItem Text="--請選擇--" Value="" />
            <asp:ListItem Text="男" Value="1" />
            <asp:ListItem Text="女" Value="0" />
        </asp:DropDownList><br />
        身高(公分)：<asp:TextBox ID="txtHeight" runat="server" /><br />
        體重(公斤)：<asp:TextBox ID="txtWeight" runat="server" /><br />

        <asp:Button ID="btnCalc" runat="server" Text="計算 BMI" OnClick="btnCalc_Click" /><br /><br />

        <!-- 驗證控制項 -->
        <asp:RequiredFieldValidator ControlToValidate="txtName"
            ErrorMessage="*姓名必填" runat="server" />
        <asp:RequiredFieldValidator ControlToValidate="ddlGender"
            InitialValue="" ErrorMessage="*性別必選" runat="server" />
        <asp:RequiredFieldValidator ControlToValidate="txtHeight"
            ErrorMessage="*身高必填" runat="server" />
        <asp:RequiredFieldValidator ControlToValidate="txtWeight"
            ErrorMessage="*體重必填" runat="server" /><br />

        <asp:RangeValidator ControlToValidate="txtHeight" MinimumValue="100" MaximumValue="300"
            Type="Integer" ErrorMessage="*身高須介於 100–300" runat="server" />
        <asp:RangeValidator ControlToValidate="txtWeight" MinimumValue="1" MaximumValue="300"
            Type="Integer" ErrorMessage="*體重須介於 1–300" runat="server" /><br />

        <asp:ValidationSummary runat="server" />
    </div>
    </form>
</body>
</html>
